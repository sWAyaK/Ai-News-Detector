# -*- coding: utf-8 -*-
"""
Провайдеры:
  - ollama   — любые локальные модели через http://localhost:11434
  - gigachat — Sber GigaChat API
  - deepseek — DeepSeek API
  - gemini   — Google AI Studio 
  - yandex   — Yandex Cloud YandexGPT API
  
Примеры запуска:
    # Локально через Ollama
    python utils/ai_generator.py \\
        --provider ollama \\
        --models llama3.1:8b mistral:7b \\
        --human-csv titles_for_generation.csv \\
        --output ai_texts_v4.csv \\
        --limit 500

    # DeepSeek API (нужен ключ из platform.deepseek.com)
    export DEEPSEEK_API_KEY="sk-..."
    python utils/ai_generator.py \\
        --provider deepseek \\
        --models deepseek-chat deepseek-reasoner \\
        --human-csv titles_for_generation.csv \\
        --output ai_texts_v4.csv \\
        --limit 250

    # Gemini (нужен ключ из aistudio.google.com)
    export GEMINI_API_KEY="..."
    python utils/ai_generator.py \\
        --provider gemini \\
        --models gemini-2.0-flash gemini-1.5-pro \\
        --human-csv titles_for_generation.csv \\
        --output ai_texts_v4.csv \\
        --limit 500

    # YandexGPT (нужны API-ключ из Yandex Cloud + Folder ID)
    export YANDEX_API_KEY="AQVN..."
    export YANDEX_FOLDER_ID="b1g..."
    python utils/ai_generator.py \\
        --provider yandex \\
        --models yandexgpt yandexgpt-lite \\
        --human-csv titles_for_generation.csv \\
        --output ai_texts_v4.csv \\
        --limit 500
"""

import argparse
import csv
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# ─── СТИЛЕВЫЕ ШАБЛОНЫ ПРОМПТА ─────────────────────────────────────────────────
# Каждый шаблон даёт чуть другой стиль; выбираются случайно при генерации.

PROMPT_STYLES = [
    {
        "name": "neutral_news",
        "system": "Ты — журналист российского информационного агентства. "
                  "Отвечай только связным текстом без разметки и без вступлений.",
        "user_template": (
            "Напиши новость на русском языке объёмом 300-500 слов на тему: {title}\n\n"
            "Стиль — нейтральный, информационный. Никакого markdown, "
            "никаких звёздочек, никаких ссылок."
        ),
        "len_range": (400, 800),  # num_predict tokens
    },
    {
        "name": "analytical",
        "system": "Ты — аналитический корреспондент.",
        "user_template": (
            "Подготовь аналитическую новостную заметку (400-600 слов) на тему: {title}\n\n"
            "Включи факты и контекст. Никакой markdown-разметки."
        ),
        "len_range": (600, 900),
    },
    {
        "name": "no_system",
        "system": None,
        "user_template": (
            "Напиши новостную статью на русском языке на тему: {title}\n\n"
            "Объём 300-500 слов, информационный стиль. Только связный текст без разметки."
        ),
        "len_range": (400, 700),
    },
    {
        "name": "reportage",
        "system": "Ты — репортёр.",
        "user_template": (
            "Сделай репортаж на тему: {title}\n\n"
            "Включи фактологию, можешь упомянуть мнения экспертов. "
            "Объём 350-550 слов. Только связный текст, никакого markdown."
        ),
        "len_range": (500, 800),
    },
    {
        "name": "concise",
        "system": None,
        "user_template": (
            "Кратко опиши новость на русском языке (200-350 слов): {title}\n\n"
            "Основные факты. Без разметки."
        ),
        "len_range": (300, 550),
    },
]


# ─── ОЧИСТКА ВЫВОДА ───────────────────────────────────────────────────────────

THINK_RE      = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
BOLD_RE       = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")
UNDERSCORE_RE = re.compile(r"_{1,2}([^_\n]+?)_{1,2}")
HEADER_RE     = re.compile(r"^#{1,6}\s+", re.MULTILINE)
MD_LINK_RE    = re.compile(r"\[([^\]]+)\]\([^)]+\)")
URL_RE        = re.compile(r"https?://\S+")
PREFIX_RE     = re.compile(r"^(текст|новость|статья|заметка)\s*[:.\-—]\s*",
                            re.IGNORECASE)
WS_RE         = re.compile(r"[ \t]+")
NEWLINES_RE   = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub("", text)            # <think>...</think> от R1
    text = BOLD_RE.sub(r"\1", text)          # **bold**, *italic*
    text = UNDERSCORE_RE.sub(r"\1", text)    # __bold__, _italic_
    text = HEADER_RE.sub("", text)           # ## заголовки
    text = MD_LINK_RE.sub(r"\1", text)       # [текст](url) → текст
    text = URL_RE.sub("", text)              # голые https://...
    text = PREFIX_RE.sub("", text.lstrip())  # "Текст:" в начале
    text = WS_RE.sub(" ", text)              # двойные пробелы
    text = NEWLINES_RE.sub("\n\n", text)     # тройные переносы
    return text.strip()


# ─── КЛИЕНТ OLLAMA ────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 240):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, model: str, system: str = None,
                 temperature: float = 0.7, num_predict: int = 800,
                 top_p: float = 0.9, repeat_penalty: float = 1.1,
                 num_ctx: int = 2048) -> str:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty,
                "num_ctx": num_ctx,        # контекст: меньше → меньше VRAM
                "num_gpu": 999,            # все слои на GPU (если влезают)
            },
        }
        if system:
            body["system"] = system
        r = requests.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()


# ─── КЛИЕНТ GIGACHAT ──────────────────────────────────────────────────────────

class GigaChatClient:
    OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API_URL   = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    def __init__(self, auth_key: str, scope: str = "GIGACHAT_API_PERS",
                 verify_ssl: bool = True, timeout: int = 60):
        self.auth_key = auth_key
        self.scope = scope
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.token = None
        self.token_expires_at = 0

    def _refresh_token(self):
        headers = {
            "Authorization": f"Basic {self.auth_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        r = requests.post(
            self.OAUTH_URL,
            headers=headers,
            data={"scope": self.scope},
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        r.raise_for_status()
        j = r.json()
        self.token = j["access_token"]
        # expires_at приходит в миллисекундах
        self.token_expires_at = j.get("expires_at", 0) / 1000

    def _ensure_token(self):
        if not self.token or time.time() > self.token_expires_at - 60:
            self._refresh_token()

    def generate(self, prompt: str, system: str = None,
                 model: str = "GigaChat", temperature: float = 0.7,
                 max_tokens: int = 900) -> str:
        self._ensure_token()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(
            self.API_URL,
            headers=headers,
            json=body,
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


# ─── КЛИЕНТ DEEPSEEK ──────────────────────────────────────────────────────────

class DeepSeekClient:
    API_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, api_key: str, timeout: int = 120):
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, model: str = "deepseek-chat",
                 system: str = None, temperature: float = 0.7,
                 max_tokens: int = 900) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(self.API_URL, headers=headers, json=body,
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


# ─── КЛИЕНТ GEMINI ────────────────────────────────────────────────────────────

class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, timeout: int = 120):
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, model: str = "gemini-2.0-flash",
                 system: str = None, temperature: float = 0.7,
                 max_tokens: int = 900) -> str:
        url = f"{self.BASE_URL}/{model}:generateContent?key={self.api_key}"

        body = {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        r = requests.post(url, json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Gemini может вернуть пустые candidates если сработал safety filter
        if not data.get("candidates"):
            raise ValueError(f"Gemini не вернул ответа: {data}")
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ─── КЛИЕНТ YANDEXGPT ─────────────────────────────────────────────────────────

class YandexGPTClient:
    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(self, api_key: str, folder_id: str, timeout: int = 120):
        self.api_key = api_key
        self.folder_id = folder_id
        self.timeout = timeout

    def generate(self, prompt: str, model: str = "yandexgpt",
                 system: str = None, temperature: float = 0.7,
                 max_tokens: int = 900) -> str:
        # Для YandexGPT modelUri имеет вид gpt://<folder_id>/<model>/latest
        model_uri = f"gpt://{self.folder_id}/{model}/latest"

        messages = []
        if system:
            messages.append({"role": "system", "text": system})
        messages.append({"role": "user", "text": prompt})

        body = {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
        }

        r = requests.post(self.API_URL, headers=headers, json=body,
                          timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Ответ: {"result": {"alternatives": [{"message": {"text": "..."}}]}}
        try:
            return data["result"]["alternatives"][0]["message"]["text"].strip()
        except (KeyError, IndexError) as e:
            raise ValueError(f"YandexGPT вернул неожиданный формат: {data}") from e


# ─── ЕДИНЫЙ ИНТЕРФЕЙС ГЕНЕРАЦИИ ───────────────────────────────────────────────

def generate_one(provider: str, model: str, title: str,
                  ollama: OllamaClient = None, giga: GigaChatClient = None,
                  deepseek: DeepSeekClient = None, gemini: GeminiClient = None,
                  yandex: YandexGPTClient = None) -> str:
    """Случайно выбирает стилевой шаблон и параметры, делает один запрос."""
    style = random.choice(PROMPT_STYLES)
    user_prompt = style["user_template"].format(title=title)
    system = style["system"]

    # рандомизация параметров для разнообразия
    temp = round(random.uniform(0.5, 0.85), 2)
    n_pred = random.randint(*style["len_range"])

    if provider == "ollama":
        text = ollama.generate(user_prompt, model=model, system=system,
                                temperature=temp, num_predict=n_pred)
    elif provider == "gigachat":
        text = giga.generate(user_prompt, system=system, model=model,
                              temperature=temp, max_tokens=n_pred)
    elif provider == "deepseek":
        text = deepseek.generate(user_prompt, model=model, system=system,
                                  temperature=temp, max_tokens=n_pred)
    elif provider == "gemini":
        text = gemini.generate(user_prompt, model=model, system=system,
                                temperature=temp, max_tokens=n_pred)
    elif provider == "yandex":
        text = yandex.generate(user_prompt, model=model, system=system,
                                temperature=temp, max_tokens=n_pred)
    else:
        raise ValueError(f"Неизвестный провайдер: {provider}")

    return clean_text(text), style["name"]


# ─── ОСНОВНОЙ ЦИКЛ ────────────────────────────────────────────────────────────

def run(args):
    titles_df = pd.read_csv(args.human_csv)
    if "title" not in titles_df.columns:
        sys.exit("В входном CSV нет колонки 'title'")
    # Берём только непустые уникальные заголовки
    titles = titles_df["title"].dropna().drop_duplicates().tolist()
    random.shuffle(titles)
    if args.limit:
        titles = titles[: args.limit * len(args.models)]  # запас на пропуски

    # Resume — читаем уже обработанное
    processed = set()
    if args.resume and os.path.exists(args.output):
        try:
            existing = pd.read_csv(args.output)
            processed = set(existing["title"].astype(str).tolist())
            print(f"Найдено {len(processed)} уже обработанных заголовков")
        except Exception as e:
            print(f"Не удалось прочитать {args.output}, начнём заново: {e}")

    # Клиенты
    ollama_client = giga_client = deepseek_client = gemini_client = yandex_client = None
    if args.provider == "ollama":
        ollama_client = OllamaClient(base_url=args.ollama_url)
    elif args.provider == "gigachat":
        key = args.gigachat_key or os.environ.get("GIGACHAT_AUTH_KEY")
        if not key:
            sys.exit("Не указан --gigachat-key и нет GIGACHAT_AUTH_KEY в env")
        giga_client = GigaChatClient(
            auth_key=key,
            scope=args.gigachat_scope,
            verify_ssl=not args.insecure,
        )
    elif args.provider == "deepseek":
        key = args.deepseek_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            sys.exit("Не указан --deepseek-key и нет DEEPSEEK_API_KEY в env")
        deepseek_client = DeepSeekClient(api_key=key)
    elif args.provider == "gemini":
        key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Не указан --gemini-key и нет GEMINI_API_KEY в env")
        gemini_client = GeminiClient(api_key=key)
    elif args.provider == "yandex":
        key = args.yandex_key or os.environ.get("YANDEX_API_KEY")
        folder = args.yandex_folder or os.environ.get("YANDEX_FOLDER_ID")
        if not key:
            sys.exit("Не указан --yandex-key и нет YANDEX_API_KEY в env")
        if not folder:
            sys.exit("Не указан --yandex-folder и нет YANDEX_FOLDER_ID в env")
        yandex_client = YandexGPTClient(api_key=key, folder_id=folder)

    # Целевые квоты по моделям
    target_per_model = args.limit if args.limit else float("inf")
    counts = {m: 0 for m in args.models}
    style_stats = {s["name"]: 0 for s in PROMPT_STYLES}

    # Открываем выход на дозапись
    write_header = not processed
    f = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["title", "text", "label", "source"])
    if write_header:
        writer.writeheader()

    pbar = tqdm(total=target_per_model * len(args.models),
                desc="Генерация AI-текстов", unit="текст")

    try:
        for title in titles:
            if all(c >= target_per_model for c in counts.values()):
                break  # лимит по всем моделям выбран
            if title in processed:
                continue

            # Балансировка: модель с минимальным count, у которой ещё есть квота
            available = [m for m in args.models if counts[m] < target_per_model]
            if not available:
                break
            model = min(available, key=lambda m: counts[m])

            try:
                text, style_name = generate_one(
                    args.provider, model, title,
                    ollama=ollama_client, giga=giga_client,
                    deepseek=deepseek_client, gemini=gemini_client,
                    yandex=yandex_client,
                )
            except Exception as e:
                tqdm.write(f"[ошибка {model}] {title[:60]}: {e}")
                time.sleep(args.delay * 2)
                continue

            # Минимальный фильтр — слишком короткие выходы пропускаем
            if not text or len(text) < 300:
                tqdm.write(f"[short  {model}] получено {len(text or '')} симв., пропуск")
                continue

            source_tag = f"ai_{args.provider}_{model.replace(':', '_').replace('/', '_')}"
            writer.writerow({
                "title": title,
                "text": text,
                "label": 1,
                "source": source_tag,
            })
            f.flush()

            counts[model] += 1
            style_stats[style_name] += 1
            processed.add(title)

            pbar.set_postfix_str(f"{model} ({counts[model]}/{target_per_model})")
            pbar.update(1)

            time.sleep(args.delay)
    finally:
        pbar.close()
        f.close()

    # Сводка
    print("\nГотово.")
    print("Сгенерировано по моделям:")
    for m, c in counts.items():
        print(f"  {m:<40} {c}")
    print("Распределение по стилям:")
    for s, c in style_stats.items():
        print(f"  {s:<20} {c}")
    print(f"Всего записей в {args.output}: {sum(counts.values())} (новых)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--provider", choices=["ollama", "gigachat", "deepseek", "gemini", "yandex"],
                    required=True, help="Откуда генерировать тексты")
    p.add_argument("--models", nargs="+", required=True,
                    help="Список моделей. ollama: теги (llama3.1:8b). "
                         "gigachat: GigaChat / GigaChat-2-Pro. "
                         "deepseek: deepseek-chat / deepseek-reasoner. "
                         "gemini: gemini-2.0-flash / gemini-1.5-pro. "
                         "yandex: yandexgpt / yandexgpt-lite")
    p.add_argument("--human-csv", required=True,
                    help="CSV с заголовками реальных новостей (колонка 'title')")
    p.add_argument("--output", required=True,
                    help="Куда дописывать сгенерированные тексты (CSV)")
    p.add_argument("--limit", type=int, default=500,
                    help="Сколько текстов на каждую модель (по умолчанию 500)")
    p.add_argument("--resume", action="store_true",
                    help="Не перегенерировать заголовки, уже встречающиеся в --output")
    p.add_argument("--delay", type=float, default=1.0,
                    help="Пауза между запросами, сек")

    # Ollama-специфичные
    p.add_argument("--ollama-url", default="http://localhost:11434")

    # GigaChat-специфичные
    p.add_argument("--gigachat-key", default=None,
                    help="Авторизационный ключ GigaChat (Basic). "
                         "Альтернатива: переменная GIGACHAT_AUTH_KEY")
    p.add_argument("--gigachat-scope", default="GIGACHAT_API_PERS",
                    choices=["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"])
    p.add_argument("--insecure", action="store_true",
                    help="Отключить проверку SSL (если не установлены сертификаты Минцифры)")

    # DeepSeek-специфичные
    p.add_argument("--deepseek-key", default=None,
                    help="API-ключ DeepSeek. Альтернатива: переменная DEEPSEEK_API_KEY")

    # Gemini-специфичные
    p.add_argument("--gemini-key", default=None,
                    help="API-ключ Google Gemini. Альтернатива: переменная GEMINI_API_KEY")

    # YandexGPT-специфичные
    p.add_argument("--yandex-key", default=None,
                    help="API-ключ Yandex Cloud. Альтернатива: переменная YANDEX_API_KEY")
    p.add_argument("--yandex-folder", default=None,
                    help="Folder ID Yandex Cloud. Альтернатива: переменная YANDEX_FOLDER_ID")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    run(args)
