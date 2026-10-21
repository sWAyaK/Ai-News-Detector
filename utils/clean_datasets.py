# -*- coding: utf-8 -*-
"""
Очистка датасетов.

Поддерживает три режима:
  --mode real       — очистка реальных новостей
  --mode ai         — очистка AI-сгенерированных текстов
  --mode combine    — объединение нескольких датасетов в один датасет

Примеры:
    python clean_datasets.py --mode real \\
        --input unpacked_dataset_real.csv \\
        --output dataset_real.csv

    python clean_datasets.py --mode ai \\
        --input generated_dataset.csv \\
        --output dataset_ai.csv

    python clean_datasets.py --mode combine \\
        --inputs dataset_real.csv dataset_ai.csv \\
        --output dataset_combined.csv \\
        --max-chars 3000
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ─── ОБЩИЕ УТИЛИТЫ ────────────────────────────────────────────────────────────

def trim_to_sentence(text, limit=3000):
    """Обрезает текст до limit символов по границе последнего полного предложения."""
    if not isinstance(text, str) or len(text) <= limit:
        return text
    chunk = text[:limit]
    last = max(
        chunk.rfind('. '),
        chunk.rfind('.\n'),
        chunk.rfind('! '),
        chunk.rfind('? '),
        chunk.rfind('."'),
        chunk.rfind('.»'),
    )
    return chunk[:last + 1].strip() if last > limit // 2 else chunk.strip()


def trim_to_complete_sentence(text):
    """Обрезает незавершённый текст до последнего полного предложения."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    if text.endswith(('.', '!', '?', '."', '.»', '…', ')', ']')):
        return text
    last = max(
        text.rfind('. '), text.rfind('! '), text.rfind('? '),
        text.rfind('.\n'), text.rfind('!\n'), text.rfind('?\n'),
    )
    return text[:last + 1].strip() if last > len(text) * 0.5 else text


def normalize_columns(df):
    """Приводит к единой схеме: title, text, label, source."""
    if 'text' in df.columns and 'text' not in df.columns:
        df = df.rename(columns={'text': 'text'})
    return df


# ─── ОЧИСТКА РЕАЛЬНЫХ НОВОСТЕЙ ────────────────────────────────────────────────

# Бойлерплейт: фразы, которыми оканчивается часть статей с сайтов
BOILERPLATE_TAILS = [
    r'\s*Этот материал показался интересным\?.*$',
    r'\s*Подписывайтесь на.*$',
    r'\s*Читайте также:.*$',
    r'\s*Поделиться в соцсетях.*$',
    r'\s*Нашли опечатку\?.*$',
    r'\s*Следите за нами в.*$',
    r'\s*Читайте нас в.*$',
    r'\s*Узнайте первыми.*$',
    r'\s*Присоединяйтесь к нам.*$',
    r'\s*Будьте в курсе.*$',
    r'\s*Получайте новости.*$',
    r'\s*Подпишитесь на.*$',
    r'\s*Подписаться на рассылку.*$',
    r'\s*Материалы по теме.*$',
    r'\s*Смотрите также.*$',
    r'\s*По теме:.*$',
]

# Шаблонный мусор в середине текста: (паттерн, замена)
NOISE_INLINE: list[tuple[re.Pattern, str]] = [
    # Авторские права
    (re.compile(r'(?i)\bвсе права защищены\b[^.!?\n]*[.!?]?'), ''),
    (re.compile(r'(?i)©\s*\d{4}[^.!?\n]*'), ''),
    # Кнопки «копировать / поделиться»
    (re.compile(r'(?i)\bскопировать ссылку\b'), ''),
    (re.compile(r'(?i)\bподелиться в соцсетях\b[^.!?\n]*[.!?]?'), ''),
    # Подписка
    (re.compile(r'(?i)\bподписаться на рассылку\b[^.!?\n]*[.!?]?'), ''),
    (re.compile(r'(?i)\bподписаться на (?:новости|обновления|telegram|телеграм)\b[^.!?\n]*[.!?]?'), ''),
    # Ссылки на соцсети
    (re.compile(
        r'(?i)\b(?:читайте нас|следите за нами|мы) в '
        r'(?:VK|ВКонтакте|Telegram|Телеграм|Twitter|Одноклассниках|'
        r'Facebook|Instagram|YouTube|Яндекс[. ]?Дзен|Дзен)\b[^.!?\n]*[.!?]?'
    ), ''),
    # Упоминания медиаресурсов как подписи (не в составе предложения)
    (re.compile(
        r'(?i)(?:^|(?<=\s)|(?<=\())'
        r'(?:lenta|ria|rbc|tass|interfax|gazeta|kommersant|rt)\.ru'
        r'(?=\s|[,).;]|$)'
    ), ''),
    # Теги
    (re.compile(r'(?im)^теги\s*:.*$'), ''),
    (re.compile(r'(?im)^рубрик[аи]\s*:.*$'), ''),
    # Подписи к фото
    (re.compile(r'(?im)^фото\s*:\s*.{1,120}$'), ''),
    (re.compile(r'(?im)^источник фото\s*:\s*.{1,120}$'), ''),
    # Реклама
    (re.compile(r'(?im)^(?:реклама|на правах рекламы)\s*$'), ''),
]


def remove_inline_noise(text: str) -> str:
    """Удаляет рекламные, сервисные и служебные фразы из текста."""
    if not isinstance(text, str):
        return text
    for pat, repl in NOISE_INLINE:
        text = pat.sub(repl, text)
    return re.sub(r' {2,}', ' ', text).strip()

def remove_duplicate_sentences(text):
    """Удаляет повторяющиеся предложения из текста, сохраняя порядок первого вхождения."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    # Разбиваем текст на предложения по .!? с пробелом или концом строки
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = set()
    unique = []
    for sent in sentences:
        sent_norm = sent.strip()
        if sent_norm and sent_norm not in seen:
            seen.add(sent_norm)
            unique.append(sent_norm)
    return ' '.join(unique)

def clean_real(input_path, output_path):
    df = pd.read_csv(input_path)
    df = normalize_columns(df)
    initial = len(df)
    print(f"Загружено реальных новостей: {initial}")

    # 1. Полные дубли по тексту
    before = len(df)
    df = df.drop_duplicates(subset='text', keep='first')
    print(f"  Полных дублей текста:    -{before - len(df)}")

    # 2. Дубли по заголовку
    before = len(df)
    df = df.drop_duplicates(subset='title', keep='first')
    print(f"  Дублей по заголовку:     -{before - len(df)}")

    # 3. Слишком короткие (по словам)
    before = len(df)
    df = df[df['text'].str.split().str.len() >= 80]
    print(f"  Слишком коротких:        -{before - len(df)}")

    # 4. Слишком короткие (по заголовкам)
    before = len(df)
    df = df[df['title'].str.split().str.len() >= 6]
    print(f"  Слишком коротких заголовков:        -{before - len(df)}")

    # 5. Не-русские
    before = len(df)
    df['_ru'] = df['text'].str.count(r'[а-яёА-ЯЁ]')
    df['_en'] = df['text'].str.count(r'[a-zA-Z]')
    df = df[df['_ru'] > df['_en']]
    df = df.drop(columns=['_ru', '_en'])
    print(f"  Не-русских:              -{before - len(df)}")

    # 6. Срезаем бойлерплейт-хвосты
    trimmed_count = 0
    for pat in BOILERPLATE_TAILS:
        compiled = re.compile(pat, re.DOTALL | re.IGNORECASE)
        was = df['text'].copy()
        df['text'] = df['text'].str.replace(compiled, '', regex=True).str.strip()
        trimmed_count += (was != df['text']).sum()
    print(f"  Обрезано хвостов-бойлерплейта: {trimmed_count}")

    # 6b. Удаляем встроенный мусор (права, соцсети, кнопки, подписки)
    before_noise = df['text'].copy()
    df['text'] = df['text'].apply(remove_inline_noise)
    noise_changed = (before_noise != df['text']).sum()
    print(f"  Очищено от встроенного мусора: {noise_changed} текстов")

    # 7. Удаляем полностью дублирующиеся предложения в каждом тексте
    before_dup_sent = df['text'].apply(lambda t: len(t.split()) if isinstance(t, str) else 0)
    df['text'] = df['text'].apply(remove_duplicate_sentences)
    after_dup_sent = df['text'].apply(lambda t: len(t.split()) if isinstance(t, str) else 0)
    removed_sent_words = (before_dup_sent - after_dup_sent).sum()
    print(f"  Удалено повторяющихся предложений (в словах): -{removed_sent_words}")

    # 8. После обрезки повторно убираем коротыши
    before = len(df)
    df = df[df['text'].str.len() >= 200]
    print(f"  Коротких после обрезки:  -{before - len(df)}")

    # Заполнить label если отсутствует
    if 'label' not in df.columns:
        df['label'] = 0
    if 'source' not in df.columns:
        df['source'] = 'real'

    df = df[['title', 'text', 'label', 'source']]
    df.to_csv(output_path, index=False)
    print(f"\nСохранено: {output_path} ({len(df)} строк, удалено {initial - len(df)})")


# ─── ОЧИСТКА AI-ТЕКСТОВ ───────────────────────────────────────────────────────

# Артефакты, которые нужно вырезать
THINK_RE      = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
BOLD_RE       = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")
UNDERSCORE_RE = re.compile(r"_{1,2}([^_\n]+?)_{1,2}")
HEADER_RE     = re.compile(r"^#{1,6}\s+", re.MULTILINE)
MD_LINK_RE    = re.compile(r"\[([^\]]+)\]\([^)]+\)")
URL_RE        = re.compile(r"https?://\S+")
PREFIX_RE     = re.compile(r"^(текст|новость|статья|заметка|вот)\s*[:.\-—]\s*",
                            re.IGNORECASE)
SELF_REF_RE   = re.compile(
    r'как (?:ИИ|искусственный интеллект|языковая модель|AI|нейросеть|LLM)',
    re.IGNORECASE
)

def strip_artifacts(text):
    """Вырезает <think>, markdown, ссылки, префиксы из вывода LLM."""
    if not isinstance(text, str):
        return text
    text = THINK_RE.sub("", text)
    text = BOLD_RE.sub(r"\1", text)
    text = UNDERSCORE_RE.sub(r"\1", text)
    text = HEADER_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("", text)
    text = PREFIX_RE.sub("", text.lstrip())
    return re.sub(r'\n{3,}', '\n\n', re.sub(r'[ \t]+', ' ', text)).strip()


def clean_ai(input_path, output_path,
             min_chars=500, en_ratio_max=0.10,
             trim_unfinished=True):
    df = pd.read_csv(input_path)
    df = normalize_columns(df)
    initial = len(df)
    print(f"Загружено AI-текстов: {initial}")

    # 1. Чистим артефакты в каждом тексте
    df['text'] = df['text'].apply(strip_artifacts)
    print("  Артефакты (markdown, <think>, URL): почищены")

    # 2. Удаляем английские/двуязычные
    df['_en'] = df['text'].str.count(r'[a-zA-Z]+').fillna(0)
    df['_ru'] = df['text'].str.count(r'[а-яёА-ЯЁ]+').fillna(0)
    df['_en_ratio'] = df['_en'] / (df['_en'] + df['_ru'] + 0.1)
    before = len(df)
    df = df[df['_en_ratio'] < en_ratio_max]
    df = df.drop(columns=['_en', '_ru', '_en_ratio'])
    print(f"  Англоязычных (>{int(en_ratio_max*100)}% en): -{before - len(df)}")

    # 3. Удаляем самоупоминания
    before = len(df)
    df = df[~df['text'].str.contains(SELF_REF_RE, na=False)]
    print(f"  Самоупоминаний AI:       -{before - len(df)}")

    # 4. Обрезаем незавершённые
    if trim_unfinished:
        df['text'] = df['text'].apply(trim_to_complete_sentence)
        print("  Незавершённые тексты обрезаны до конца предложения")

    # 5. Слишком короткие
    before = len(df)
    df = df[df['text'].str.len() >= min_chars]
    print(f"  Короче {min_chars} симв.:        -{before - len(df)}")

    # 6. Дубли по тексту
    before = len(df)
    df = df.drop_duplicates(subset='text', keep='first')
    print(f"  Дублей по тексту:        -{before - len(df)}")

    # 7. Дубли по (title, source)
    if 'title' in df.columns and 'source' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['title', 'source'], keep='first')
        print(f"  Дублей (title, source):  -{before - len(df)}")

    if 'label' not in df.columns:
        df['label'] = 1

    cols_out = ['title', 'text', 'label', 'source']
    df = df[[c for c in cols_out if c in df.columns]]
    df.to_csv(output_path, index=False)
    print(f"\nСохранено: {output_path} ({len(df)} строк, удалено {initial - len(df)})")


# ─── ОБЪЕДИНЕНИЕ ──────────────────────────────────────────────────────────────

def combine(input_paths, output_path, max_chars=3000, min_chars=500):
    parts = []
    for p in input_paths:
        df = pd.read_csv(p)
        df = normalize_columns(df)
        if 'label' not in df.columns:
            sys.exit(f"Нет колонки label в {p}")
        for c in ('title', 'text', 'label', 'source'):
            if c not in df.columns:
                df[c] = '' if c != 'label' else 0
        parts.append(df[['title', 'text', 'label', 'source']])
        print(f"  {p}: {len(df)} строк, label values = {sorted(df['label'].unique())}")

    combined = pd.concat(parts, ignore_index=True)
    print(f"\nОбъединено: {len(combined)}")

    # Обрезка до max_chars
    print(f"Обрезаем тексты до {max_chars} символов по границе предложения...")
    combined['text'] = combined['text'].apply(
        lambda t: trim_to_sentence(t, limit=max_chars))

    # Дубли
    before = len(combined)
    combined = combined.drop_duplicates(subset='text', keep='first')
    print(f"  Дубли:                   -{before - len(combined)}")

    # Минимальная длина
    before = len(combined)
    combined = combined[combined['text'].str.len() >= min_chars]
    print(f"  Короче {min_chars} симв.:        -{before - len(combined)}")

    # Перемешиваем
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print(f"  всего:           {len(combined)}")
    print(f"  label=0 (real):  {(combined['label']==0).sum()}")
    print(f"  label=1 (AI):    {(combined['label']==1).sum()}")
    print(f"\nРаспределение по источникам:")
    print(combined['source'].value_counts().to_string())
    print(f"Итоговый размер датасета: ", len(combined))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    p.add_argument('--mode', choices=['real', 'ai', 'combine'], required=True)
    p.add_argument('--input', help='Один входной файл (для режимов real/ai)')
    p.add_argument('--inputs', nargs='+',
                    help='Несколько входных файлов (для режима combine)')
    p.add_argument('--output', required=True)
    p.add_argument('--min-chars', type=int, default=500,
                    help='Минимальная длина текста (default: 500)')
    p.add_argument('--max-chars', type=int, default=3000,
                    help='Обрезка по длине в режиме combine (default: 3000)')
    p.add_argument('--en-ratio-max', type=float, default=0.10,
                    help='Порог доли английских слов (default: 0.10)')

    args = p.parse_args()

    if args.mode == 'real':
        if not args.input:
            sys.exit('Для --mode real нужен --input')
        clean_real(args.input, args.output)
    elif args.mode == 'ai':
        if not args.input:
            sys.exit('Для --mode ai нужен --input')
        clean_ai(args.input, args.output,
                 min_chars=args.min_chars,
                 en_ratio_max=args.en_ratio_max)
    elif args.mode == 'combine':
        if not args.inputs:
            sys.exit('Для --mode combine нужен --inputs')
        combine(args.inputs, args.output,
                max_chars=args.max_chars,
                min_chars=args.min_chars)


if __name__ == '__main__':
    main()
