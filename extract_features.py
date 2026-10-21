"""
Извлечение лемматических и стилистических признаков для детекции AI-текстов.
Не требует pymorphy2 — использует суффиксный анализ и маркерные паттерны.
"""

import re
import pandas as pd
import numpy as np
from collections import Counter

# ── Стоп-слова (функциональные слова русского языка) ──────────────────────────
STOP_WORDS = {
    'и','в','во','не','что','он','на','я','с','со','как','а','то','все','она',
    'так','его','но','да','ты','к','у','же','вы','за','бы','по','только','ее',
    'мне','было','вот','от','меня','еще','нет','о','из','ему','теперь','когда',
    'даже','ну','вдруг','ли','если','уже','или','ни','быть','был','него',
    'до','вас','нибудь','опять','уж','вам','ведь','там','потом','себя','ничего',
    'ей','может','они','тут','где','есть','надо','ней','для','мы','тебя','их',
    'чем','была','сам','чтоб','без','будто','чего','раз','тоже','себе','под',
    'будет','ж','тогда','кто','этот','того','потому','этого','какой','совсем',
    'ним','здесь','этом','один','почти','мой','тем','чтобы','нее','сейчас',
    'были','куда','зачем','всех','никогда','можно','при','наконец','два',
    'об','другой','хоть','после','над','больше','тот','через','эти','нас',
    'про','всего','них','какая','много','разве','три','эту','моя','впрочем',
    'хорошо','свою','этой','перед','иногда','лучше','чуть','том','нельзя',
    'такой','им','более','всегда','конечно','всю','между','это','при',
}

# ── AI-маркерные леммы и фразы ────────────────────────────────────────────────
AI_MARKERS = [
    r'в заключени',      # "в заключение", "в заключении"
    r'таким образом',
    r'следует отметить',
    r'необходимо отметить',
    r'следует подчеркнуть',
    r'необходимо подчеркнуть',
    r'подводя итог',
    r'подводя итоги',
    r'в целом',
    r'таким образом',
    r'важно отметить',
    r'важно подчеркнуть',
    r'можно сказать',
    r'можно сделать вывод',
    r'следует сказать',
    r'прежде всего',
    r'во-первых',
    r'во-вторых',
    r'в-третьих',
    r'с одной стороны',
    r'с другой стороны',
    r'кроме того',
    r'помимо этого',
    r'при этом',
    r'в связи с этим',
    r'исходя из',
    r'на основании',
    r'таким образом',
    r'в результате',
    r'в итоге',
    r'в данном контексте',
    r'в данном случае',
    r'данный вопрос',
    r'данная проблема',
    r'данная тема',
    r'является важным',
    r'является ключевым',
    r'играет важную роль',
    r'играет ключевую роль',
]

# ── Морфологические суффиксные паттерны ───────────────────────────────────────
# Прилагательные
ADJ_SUFFIXES = re.compile(
    r'(ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ым|им|ой|ей|ую|юю)$'
)
# Наречия
ADV_SUFFIXES = re.compile(
    r'(льно|ально|ически|ски|ово|ево|ажно|ожно|ично|ачно)$'
)
# Глаголы (инфинитив и спряжения)
VERB_SUFFIXES = re.compile(
    r'(ть|ться|тся|ет|ит|ют|ают|уют|ешь|ишь|ем|им|ете|ите|ют|ал|ала|али|ило|ился|илась|ились)$'
)
# Пассивные конструкции
PASSIVE_SUFFIXES = re.compile(
    r'(ется|ются|ован|ована|ованы|ирован|ирована|ированы|нный|нная|нные|нного)$'
)
# Существительные-абстракции (характерны для AI)
ABSTRACT_NOUN_SUFFIXES = re.compile(
    r'(ость|ность|ение|ание|ство|ация|изация|ификация|ирование)$'
)


def tokenize(text: str) -> list:
    """Разбить на кириллические слова."""
    return re.findall(r'[а-яёА-ЯЁ]+', text.lower())


def sent_tokenize(text: str) -> list:
    """Грубое разбиение на предложения."""
    sents = re.split(r'[.!?]+', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def extract_features(text: str) -> dict:
    if not isinstance(text, str) or len(text) < 50:
        return {}

    tokens = tokenize(text)
    if len(tokens) < 10:
        return {}

    content_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    sentences = sent_tokenize(text)
    n = len(tokens)
    n_sent = max(len(sentences), 1)
    n_unique = len(set(tokens))
    n_content = len(content_tokens)

    feat = {}

    # ── 1. БАЗОВАЯ СТАТИСТИКА ─────────────────────────────────────────────────
    feat['text_len']          = len(text)
    feat['word_count']        = n
    feat['sent_count']        = n_sent
    feat['avg_sent_len']      = n / n_sent
    feat['avg_word_len']      = np.mean([len(t) for t in tokens]) if tokens else 0
    feat['max_sent_len']      = max((len(tokenize(s)) for s in sentences), default=0)

    # ── 2. ЛЕКСИЧЕСКОЕ РАЗНООБРАЗИЕ ───────────────────────────────────────────
    feat['ttr']               = n_unique / n if n > 0 else 0
    feat['content_ratio']     = n_content / n if n > 0 else 0
    feat['stopword_ratio']    = 1 - feat['content_ratio']

    # Hapax legomena (слова, встречающиеся ровно 1 раз)
    freq = Counter(tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    feat['hapax_ratio']       = hapax / n_unique if n_unique > 0 else 0

    # MTLD-приближение (упрощённый вариант через скользящее окно)
    window = 50
    if n >= window:
        ttrs = [len(set(tokens[i:i+window])) / window
                for i in range(0, n - window, window // 2)]
        feat['mtld_approx']   = float(np.mean(ttrs))
    else:
        feat['mtld_approx']   = feat['ttr']

    # ── 3. МОРФОЛОГИЧЕСКИЙ ПРОФИЛЬ (суффиксы) ────────────────────────────────
    adj_c = adv_c = verb_c = pass_c = abst_c = 0
    for t in content_tokens:
        if ADJ_SUFFIXES.search(t):      adj_c  += 1
        if ADV_SUFFIXES.search(t):      adv_c  += 1
        if VERB_SUFFIXES.search(t):     verb_c += 1
        if PASSIVE_SUFFIXES.search(t):  pass_c += 1
        if ABSTRACT_NOUN_SUFFIXES.search(t): abst_c += 1

    denom = n_content if n_content > 0 else 1
    feat['adj_ratio']      = adj_c  / denom
    feat['adv_ratio']      = adv_c  / denom
    feat['verb_ratio']     = verb_c / denom
    feat['passive_ratio']  = pass_c / denom
    feat['abstract_ratio'] = abst_c / denom

    # ── 4. AI-МАРКЕРНЫЕ ФРАЗЫ ────────────────────────────────────────────────
    text_lower = text.lower()
    marker_hits = sum(1 for m in AI_MARKERS if re.search(m, text_lower))
    feat['ai_marker_count'] = marker_hits
    feat['ai_marker_density'] = marker_hits / n_sent

    # Конкретные высокоинформативные маркеры
    feat['has_conclusion']   = int(bool(re.search(r'в заключени', text_lower)))
    feat['has_takim']        = int(bool(re.search(r'таким образом', text_lower)))
    feat['has_sleduet']      = int(bool(re.search(r'следует (отметить|подчеркнуть|сказать)', text_lower)))
    feat['has_pervoedvoe']   = int(bool(re.search(r'во-первых|во-вторых', text_lower)))
    feat['has_dannyi']       = int(bool(re.search(r'данн(ый|ая|ое|ые|ого|ому|ым|ой|ых|ым)', text_lower)))
    feat['has_igraet_rol']   = int(bool(re.search(r'играет (важную|ключевую|значительную) роль', text_lower)))

    # ── 5. ПУНКТУАЦИЯ И СТРУКТУРА ─────────────────────────────────────────────
    feat['comma_ratio']      = text.count(',')  / n if n > 0 else 0
    feat['quote_ratio']      = (text.count('«') + text.count('"')) / n if n > 0 else 0
    feat['colon_ratio']      = text.count(':')  / n if n > 0 else 0
    feat['dash_ratio']       = (text.count('—') + text.count('-')) / n if n > 0 else 0

    # Числа и даты (маркер реальных новостей)
    feat['digit_token_ratio'] = len(re.findall(r'\b\d+\b', text)) / n if n > 0 else 0

    # Заглавные слова в середине предложения (имена/организации — маркер реальных новостей)
    caps_mid = len(re.findall(r'(?<=[а-яё] )[А-ЯЁ][а-яё]+', text))
    feat['proper_noun_approx'] = caps_mid / n if n > 0 else 0

    # ── 6. ЛЕКСИЧЕСКИЕ БИГРАММЫ-МАРКЕРЫ ──────────────────────────────────────
    bigrams = list(zip(tokens, tokens[1:]))
    # Клише-биграммы реальных новостей
    news_bigrams = {('по', 'данным'), ('в', 'ходе'), ('в', 'рамках'),
                    ('в', 'связи'), ('в', 'том'), ('в', 'соответствии'),
                    ('на', 'заседании'), ('в', 'отношении')}
    # Клише-биграммы AI-текстов
    ai_bigrams = {('является', 'важным'), ('необходимо', 'отметить'),
                  ('следует', 'отметить'), ('можно', 'сказать'),
                  ('данного', 'вопроса'), ('данной', 'проблемы'),
                  ('в', 'целом'), ('кроме', 'того'), ('при', 'этом'),
                  ('помимо', 'этого'), ('на', 'основании')}
    bigram_set = set(bigrams)
    feat['news_bigram_hits'] = sum(1 for b in news_bigrams if b in bigram_set)
    feat['ai_bigram_hits']   = sum(1 for b in ai_bigrams   if b in bigram_set)

    return feat
