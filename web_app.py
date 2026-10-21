"""
NewsDetector Web Application - Flask backend
"""
import json
import re
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import requests
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup

from app.predict import predict_text, extract_features

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / 'model_gb.pkl'
with open(MODEL_PATH, 'rb') as _f:
    _bundle = pickle.load(_f)
MODEL        = _bundle['model']
FEATURE_COLS = _bundle['feature_cols']

# ── News sources ──────────────────────────────────────────────────────────────
RSS_FEEDS = [
    {'id': 'lenta',       'name': 'Lenta.ru',         'url': 'https://lenta.ru/rss/news'},
    {'id': 'ria',         'name': 'РИА Новости',      'url': 'https://ria.ru/export/rss2/archive/index.xml'},
    {'id': 'tass',        'name': 'ТАСС',              'url': 'https://tass.ru/rss/v2.xml'},
    {'id': 'rbc',         'name': 'RBC',               'url': 'https://rssexport.rbc.ru/rbcnews/news/30/full.rss'},
    {'id': 'gazeta_pol',  'name': 'Газета.ru — Полит', 'url': 'https://www.gazeta.ru/export/rss/politics.xml'},
    {'id': 'gazeta_soc',  'name': 'Газета.ru — Общ',  'url': 'https://www.gazeta.ru/export/rss/social.xml'},
    {'id': 'interfax',    'name': 'Интерфакс',         'url': 'https://www.interfax.ru/rss.asp'},
    {'id': 'kommersant',  'name': 'Коммерсантъ',       'url': 'https://www.kommersant.ru/RSS/main.xml'},
]

HEADERS = {
    'User-Agent':                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept':                    'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language':           'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding':           'gzip, deflate',
    'Connection':                'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Дополнительные заголовки для конкретных сайтов (Referer снимает часть 403)
SITE_EXTRA_HEADERS: dict[str, dict] = {
    'tass.ru':    {'Referer': 'https://tass.ru/'},
    'rbc.ru':     {'Referer': 'https://www.rbc.ru/'},
    'gazeta.ru':  {'Referer': 'https://www.gazeta.ru/'},
    'interfax.ru':{'Referer': 'https://www.interfax.ru/'},
}

SITE_SELECTORS = {
    'lenta.ru':      ['.topic-page__content', '.js-topic__text',
                      '.b_article-content', '[class*="topic-body"]',
                      '[class="topic-body__title"]'],
    'ria.ru':        ['.article__body', '.article__text',
                      '[class*="article-body"]', '[itemprop="articleBody"]',
                      '[class="article__title"]'],
    'tass.ru':       ['.text-content', '.news__text',
                      '[class*="NewsContent"]', '[class*="news-article"]',
                      '[class="ContentPageContainer_container__25Wpb"]',
                      '[class="MaterialPageTitle_wrapper__rxyPX"]',
                      '[class="Paragraph_paragraph__Sxf1_"]'],
    'kommersant.ru': ['.article__text__content', '.article_body',
                      '[itemprop="articleBody"]'],
    'rbc.ru':        ['.article__content', '.article__text',
                      '.article-body__content', '[class*="article__body"]'],
    'gazeta.ru':     ['.b-text', '.article_full_text',
                      '[itemprop="articleBody"]', '[class*="article-text"]',
                      '[class="b_article-title"]',
                      '[class="b_article-intro"]',
                      '[class="b_article-text "]'],
    'interfax.ru':   ['#news_text', '.news_text',
                      '.textContent', '[class*="news-body"]'],
}

# Служебные фразы, которые нужно вырезать из извлечённого текста
_BOILERPLATE_WEB: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?i)\bвсе права защищены\b[^.!?\n]*[.!?]?'), ''),
    (re.compile(r'(?i)©\s*\d{4}[^.!?\n]*'), ''),
    (re.compile(r'(?i)\bскопировать ссылку\b'), ''),
    (re.compile(r'(?i)\bподелиться в соцсетях\b[^.!?\n]*[.!?]?'), ''),
    (re.compile(r'(?i)\bподелиться:?\s*$', re.MULTILINE), ''),
    (re.compile(r'(?i)\bподписаться на рассылку\b[^.!?\n]*[.!?]?'), ''),
    (re.compile(r'(?i)\bподписаться на (?:новости|обновления|telegram|телеграм)\b[^.!?\n]*[.!?]?'), ''),
    (re.compile(
        r'(?i)\b(?:читайте нас|следите за нами|мы) в '
        r'(?:VK|ВКонтакте|Telegram|Телеграм|Twitter|Одноклассниках|'
        r'Facebook|Instagram|YouTube|Яндекс[. ]?Дзен|Дзен)\b[^.!?\n]*[.!?]?'
    ), ''),
    (re.compile(
        r'(?i)(?:^|(?<=\s)|(?<=\())'
        r'(?:lenta|ria|rbc|tass|interfax|gazeta|kommersant|rt)\.ru'
        r'(?=\s|[,).;]|$)'
    ), ''),
    (re.compile(r'(?im)^теги\s*:.*$'), ''),
    (re.compile(r'(?im)^фото\s*:\s*.{1,120}$'), ''),
    (re.compile(r'(?im)^(?:реклама|на правах рекламы)\s*$'), ''),
]

# Признаки SPA/JS-рендеринга (React, Next.js, Vue, Nuxt)
_SPA_MARKERS = [
    'window.__NEXT_DATA__',
    'window.__INITIAL_STATE__',
    '__NUXT__',
    'window.initialData',
    '__remixContext',
    'window.__APP_STATE__',
    '<div id="__next">',
    '<div id="root">',
    '<div id="app">',
    '<div id="react-app">',
]

MARKER_LABELS = {
    'has_conclusion': '«В заключение»',
    'has_takim':      '«Таким образом»',
    'has_sleduet':    '«Следует отметить»',
    'has_pervoedvoe': '«Во-первых / Во-вторых»',
    'has_dannyi':     '«Данный / данная»',
    'has_igraet_rol': '«Играет роль»',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_web_text(text: str) -> str:
    """Вырезает служебные фразы из текста, извлечённого с веб-страницы."""
    for pat, repl in _BOILERPLATE_WEB:
        text = pat.sub(repl, text)
    return re.sub(r' {2,}', ' ', text).strip()


def _extract_jsonld(soup: BeautifulSoup) -> str:
    """Пытается извлечь тело статьи из JSON-LD (schema.org ArticleBody)."""
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '{}')
            if isinstance(data, list):
                data = data[0] if data else {}
            body = data.get('articleBody') or data.get('description') or ''
            if len(body) > 200:
                return body
        except Exception:
            continue
    return ''


def _entry_fulltext(e) -> str:
    """Извлекает полный текст из <content:encoded> (RBC full.rss и аналоги)."""
    for c in e.get('content', []):
        v = re.sub(r'<[^>]+>', '', c.get('value', '')).strip()
        # Проверяем, что это кириллица, а не бинарные данные
        if len(v) > 150 and re.search(r'[а-яёА-ЯЁ]', v):
            return v
    return ''


def _fetch_feed(feed_info: dict) -> list:
    try:
        feed = feedparser.parse(feed_info['url'])
        out = []
        for e in feed.entries[:30]:
            summary   = re.sub(r'<[^>]+>', '', e.get('summary', '')).strip()
            rss_text = _entry_fulltext(e)
            out.append({
                'title':     e.get('title', '').strip(),
                'summary':   (summary or rss_text[:280])[:280],
                'text':      rss_text[:4000],
                'link':      e.get('link', ''),
                'source':    feed_info['name'],
                'source_id': feed_info['id'],
                'published': e.get('published', ''),
            })
        return out
    except Exception:
        return []


def _extract_text(html: str, url: str = '') -> str:
    soup = BeautifulSoup(html, 'html.parser')

    # 1. JSON-LD (schema.org) — наиболее надёжный источник для новостных сайтов
    jsonld = _extract_jsonld(soup)
    if jsonld:
        return _clean_web_text(jsonld)

    for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                     'aside', 'iframe', 'noscript', 'form', 'button']):
        tag.decompose()

    # 2. Специфичные для сайта селекторы
    for site, selectors in SITE_SELECTORS.items():
        if site in url:
            for sel in selectors:
                try:
                    el = soup.select_one(sel)
                    if el:
                        t = el.get_text(' ', strip=True)
                        if len(t) > 100:
                            return _clean_web_text(t)
                except Exception:
                    continue

    # 3. Универсальные семантические селекторы
    for sel in [
        '[itemprop="articleBody"]',
        'article',
        '[class*="article-body"]',  '[class*="article__body"]',
        '[class*="articleBody"]',   '[class*="article__content"]',
        '[class*="news__text"]',    '[class*="news-text"]',
        '[class*="newsText"]',      '[class*="news-body"]',
        '[class*="post-content"]',  '[class*="entry-content"]',
        '[class*="content__text"]', '[class*="text__content"]',
        '.text', '#text', '#content', '#article',
        'main',
    ]:
        try:
            el = soup.select_one(sel)
            if el:
                t = el.get_text(' ', strip=True)
                if len(t) > 100:
                    return _clean_web_text(t)
        except Exception:
            continue

    # 4. Все <p> длиннее 40 символов в порядке документа
    paras = [p.get_text(strip=True) for p in soup.find_all('p')
             if len(p.get_text(strip=True)) > 40]
    if paras:
        return _clean_web_text(' '.join(paras))

    # 5. Последний резерв: весь видимый текст страницы
    return _clean_web_text(soup.get_text(' ', strip=True))


def _build_result(text: str):
    pred, p_human, p_ai = predict_text(text, MODEL, FEATURE_COLS)
    if pred is None:
        return None
    feat    = extract_features(text)
    markers = [v for k, v in MARKER_LABELS.items() if feat.get(k, 0)]
    return {
        'label':   'AI' if p_ai >= 0.5 else 'ЧЕЛОВЕК',
        'p_ai':    round(float(p_ai), 4),
        'p_human': round(float(p_human), 4),
        'markers': markers,
        'stats': {
            'word_count':        int(feat.get('word_count', 0)),
            'sent_count':        int(feat.get('sent_count', 0)),
            'ttr':               round(float(feat.get('ttr', 0)), 3),
            'ai_marker_count':   int(feat.get('ai_marker_count', 0)),
            'ai_marker_density': round(float(feat.get('ai_marker_density', 0)), 3),
            'abstract_ratio':    round(float(feat.get('abstract_ratio', 0)), 3),
            'passive_ratio':     round(float(feat.get('passive_ratio', 0)), 3),
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/news')
def api_news():
    query = request.args.get('q', '').strip().lower()
    srcs  = set(request.args.getlist('src')) or {f['id'] for f in RSS_FEEDS}
    feeds = [f for f in RSS_FEEDS if f['id'] in srcs]

    articles = []
    with ThreadPoolExecutor(max_workers=max(len(feeds), 1)) as ex:
        futures = [ex.submit(_fetch_feed, f) for f in feeds]
        try:
            for fut in as_completed(futures, timeout=15):
                try:
                    articles.extend(fut.result())
                except Exception:
                    pass
        except TimeoutError:
            # Забираем результаты уже завершённых фьючерсов, остальные игнорируем
            for fut in futures:
                if fut.done():
                    try:
                        articles.extend(fut.result())
                    except Exception:
                        pass

    if query:
        articles = [a for a in articles
                    if query in a['title'].lower() or query in a['summary'].lower()]
        articles.sort(key=lambda a: query not in a['title'].lower())

    return jsonify(articles[:60])


@app.route('/api/fetch_url', methods=['POST'])
def api_fetch_url():
    data = request.get_json(silent=True) or {}
    url  = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'URL не указан'}), 400
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Неверный формат URL'}), 400

    try:
        req_headers = dict(HEADERS)
        for site, extra in SITE_EXTRA_HEADERS.items():
            if site in url:
                req_headers.update(extra)
                break

        resp = requests.get(url, headers=req_headers, timeout=15, allow_redirects=True)

        if resp.status_code in (401, 403):
            return jsonify({
                'error': f'Сайт отказал в доступе (HTTP {resp.status_code}). '
                         f'Сайт блокирует автоматические запросы. '
                         f'Скопируйте текст статьи вручную и вставьте в поле «Свой текст».'
            }), 400

        # Определяем кодировку: Content-Type > chardet > utf-8
        ct    = resp.headers.get('content-type', '')
        ct_m  = re.search(r'charset=([^\s;]+)', ct, re.IGNORECASE)
        if ct_m:
            resp.encoding = ct_m.group(1).strip()
        elif resp.apparent_encoding and resp.apparent_encoding.upper() not in ('ASCII', 'WINDOWS-1252', 'ISO-8859-1'):
            resp.encoding = resp.apparent_encoding
        else:
            resp.encoding = 'utf-8'

        soup  = BeautifulSoup(resp.text, 'html.parser')
        title = (soup.title.string or '').strip() if soup.title else ''
        text  = re.sub(r'\s+', ' ', _extract_text(resp.text, url)).strip()

        if len(text) < 50:
            if any(m in resp.text for m in _SPA_MARKERS):
                return jsonify({
                    'error': 'Страница использует клиентский рендеринг (React/Next.js). '
                             'Текст недоступен без браузера. '
                             'Скопируйте текст статьи вручную и вставьте в поле «Свой текст».'
                }), 400
            return jsonify({
                'error': f'Не удалось извлечь текст статьи '
                         f'(получено {len(text)} символов). '
                         f'Сайт может блокировать парсинг или требовать авторизации.'
            }), 400

        return jsonify({'text': text[:8000], 'title': title})
    except requests.Timeout:
        return jsonify({'error': 'Таймаут соединения (15 с)'}), 400
    except requests.exceptions.SSLError:
        return jsonify({'error': 'Ошибка SSL-сертификата сайта'}), 400
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Не удалось подключиться к сайту'}), 400
    except Exception as e:
        return jsonify({'error': f'Ошибка загрузки: {str(e)[:200]}'}), 400


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    if len(text) < 50:
        return jsonify({'error': 'Текст слишком короткий (минимум 50 символов)'}), 400

    res = _build_result(text)
    if res is None:
        return jsonify({'error': 'Не удалось проанализировать (слишком мало слов)'}), 400

    return jsonify(res)


@app.route('/api/analyze_file', methods=['POST'])
def api_analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не предоставлен'}), 400

    f = request.files['file']
    if not f.filename or Path(f.filename).suffix.lower() not in ('.txt', '.md', '.text'):
        return jsonify({'error': 'Поддерживаются только .txt файлы'}), 400

    raw = f.read()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('cp1251', errors='replace')

    text = text.strip()
    if len(text) < 50:
        return jsonify({'error': 'Файл слишком короткий'}), 400

    res = _build_result(text)
    if res is None:
        return jsonify({'error': 'Не удалось проанализировать'}), 400

    res['text_preview'] = text[:600]
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
