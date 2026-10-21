"""
Детектор AI-новостей — инференс-скрипт.

Использование:
    python predict.py текст.txt
    python predict.py текст.txt --verbose
    python predict.py папка/*.txt
"""

import sys
import re
import pickle
import argparse
import os
from pathlib import Path
from collections import Counter
import numpy as np

# ── Воспроизводим feature extractor из extract_features.py ───────────────────

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

AI_MARKERS = [
    r'в заключени', r'таким образом', r'следует отметить',
    r'необходимо отметить', r'следует подчеркнуть', r'необходимо подчеркнуть',
    r'подводя итог', r'подводя итоги', r'в целом', r'важно отметить',
    r'важно подчеркнуть', r'можно сказать', r'можно сделать вывод',
    r'следует сказать', r'прежде всего', r'во-первых', r'во-вторых',
    r'в-третьих', r'с одной стороны', r'с другой стороны', r'кроме того',
    r'помимо этого', r'при этом', r'в связи с этим', r'исходя из',
    r'на основании', r'в результате', r'в итоге', r'в данном контексте',
    r'в данном случае', r'данный вопрос', r'данная проблема', r'данная тема',
    r'является важным', r'является ключевым', r'играет важную роль',
    r'играет ключевую роль',
]

ADJ_SUFFIXES     = re.compile(r'(ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ым|им|ой|ей|ую|юю)$')
ADV_SUFFIXES     = re.compile(r'(льно|ально|ически|ски|ово|ево|ажно|ожно|ично|ачно)$')
VERB_SUFFIXES    = re.compile(r'(ть|ться|тся|ет|ит|ют|ают|уют|ешь|ишь|ем|им|ете|ите|ют|ал|ала|али|ило|ился|илась|ились)$')
PASSIVE_SUFFIXES = re.compile(r'(ется|ются|ован|ована|ованы|ирован|ирована|ированы|нный|нная|нные|нного)$')
ABSTRACT_NOUN    = re.compile(r'(ость|ность|ение|ание|ство|ация|изация|ификация|ирование)$')


def tokenize(text):
    return re.findall(r'[а-яёА-ЯЁ]+', text.lower())

def sent_tokenize(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]

def extract_features(text):
    tokens = tokenize(text)
    if len(tokens) < 10:
        return None
    content = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    sentences = sent_tokenize(text)
    n, n_sent = len(tokens), max(len(sentences), 1)
    n_unique, n_content = len(set(tokens)), len(content)

    feat = {}
    feat['text_len']       = len(text)
    feat['word_count']     = n
    feat['sent_count']     = n_sent
    feat['avg_sent_len']   = n / n_sent
    feat['avg_word_len']   = np.mean([len(t) for t in tokens])
    feat['max_sent_len']   = max((len(tokenize(s)) for s in sentences), default=0)
    feat['ttr']            = n_unique / n
    feat['content_ratio']  = n_content / n
    feat['stopword_ratio'] = 1 - feat['content_ratio']

    freq   = Counter(tokens)
    hapax  = sum(1 for v in freq.values() if v == 1)
    feat['hapax_ratio']    = hapax / n_unique if n_unique > 0 else 0

    window = 50
    if n >= window:
        ttrs = [len(set(tokens[i:i+window]))/window for i in range(0, n-window, window//2)]
        feat['mtld_approx'] = float(np.mean(ttrs))
    else:
        feat['mtld_approx'] = feat['ttr']

    adj_c = adv_c = verb_c = pass_c = abst_c = 0
    for t in content:
        if ADJ_SUFFIXES.search(t):   adj_c  += 1
        if ADV_SUFFIXES.search(t):   adv_c  += 1
        if VERB_SUFFIXES.search(t):  verb_c += 1
        if PASSIVE_SUFFIXES.search(t): pass_c += 1
        if ABSTRACT_NOUN.search(t):  abst_c += 1

    d = n_content if n_content > 0 else 1
    feat['adj_ratio']      = adj_c  / d
    feat['adv_ratio']      = adv_c  / d
    feat['verb_ratio']     = verb_c / d
    feat['passive_ratio']  = pass_c / d
    feat['abstract_ratio'] = abst_c / d

    tl = text.lower()
    hits = sum(1 for m in AI_MARKERS if re.search(m, tl))
    feat['ai_marker_count']   = hits
    feat['ai_marker_density'] = hits / n_sent
    feat['has_conclusion']    = int(bool(re.search(r'в заключени', tl)))
    feat['has_takim']         = int(bool(re.search(r'таким образом', tl)))
    feat['has_sleduet']       = int(bool(re.search(r'следует (отметить|подчеркнуть|сказать)', tl)))
    feat['has_pervoedvoe']    = int(bool(re.search(r'во-первых|во-вторых', tl)))
    feat['has_dannyi']        = int(bool(re.search(r'данн(ый|ая|ое|ые|ого|ому|ым|ой|ых|ым)', tl)))
    feat['has_igraet_rol']    = int(bool(re.search(r'играет (важную|ключевую|значительную) роль', tl)))

    feat['comma_ratio']        = text.count(',')  / n
    feat['quote_ratio']        = (text.count('«') + text.count('"')) / n
    feat['colon_ratio']        = text.count(':')  / n
    feat['dash_ratio']         = (text.count('—') + text.count('-')) / n
    feat['digit_token_ratio']  = len(re.findall(r'\b\d+\b', text)) / n
    feat['proper_noun_approx'] = len(re.findall(r'(?<=[а-яё] )[А-ЯЁ][а-яё]+', text)) / n

    bigrams = list(zip(tokens, tokens[1:]))
    bigram_set = set(bigrams)
    news_b = {('по','данным'),('в','ходе'),('в','рамках'),('в','связи'),
              ('в','том'),('в','соответствии'),('на','заседании'),('в','отношении')}
    ai_b   = {('является','важным'),('необходимо','отметить'),('следует','отметить'),
              ('можно','сказать'),('данного','вопроса'),('данной','проблемы'),
              ('в','целом'),('кроме','того'),('при','этом'),
              ('помимо','этого'),('на','основании')}
    feat['news_bigram_hits'] = sum(1 for b in news_b if b in bigram_set)
    feat['ai_bigram_hits']   = sum(1 for b in ai_b   if b in bigram_set)

    return feat


def predict_text(text, model, feature_cols, verbose=False):
    feat = extract_features(text)
    if feat is None:
        return None, None, None

    import pandas as pd
    X = pd.DataFrame([feat])[feature_cols].fillna(0)
    pred   = model.predict(X)[0]
    proba  = model.predict_proba(X)[0]

    if verbose:
        print("\n── Сработавшие маркеры ──────────────────────────────")
        marker_feat = {
            'has_conclusion': '«В заключение»',
            'has_takim':      '«Таким образом»',
            'has_sleduet':    '«Следует отметить»',
            'has_pervoedvoe': '«Во-первых / Во-вторых»',
            'has_dannyi':     '«Данный/данная»',
            'has_igraet_rol': '«Играет роль»',
        }
        found = [label for key, label in marker_feat.items() if feat.get(key, 0)]
        if found:
            for f in found:
                print(f"  + {f}")
        else:
            print("  (маркерных фраз не найдено)")

        print("\n── Ключевые метрики текста ──────────────────────────")
        print(f"  Слов:               {feat['word_count']}")
        print(f"  Предложений:        {feat['sent_count']}")
        print(f"  TTR:                {feat['ttr']:.3f}")
        print(f"  Доля цифр:          {feat['digit_token_ratio']:.3f}")
        print(f"  Доля прилагательных:{feat['adj_ratio']:.3f}")
        print(f"  Абстрактные сущ.:   {feat['abstract_ratio']:.3f}")
        print(f"  Плотность AI-фраз:  {feat['ai_marker_density']:.3f}")

    return pred, proba[0], proba[1]


def main():
    parser = argparse.ArgumentParser(description='Детектор AI-новостей')
    parser.add_argument('files', nargs='+', help='Путь к текстовым файлам (.txt)')
    parser.add_argument('--model', default='model_gb.pkl', help='Путь к модели (default: model_gb.pkl)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Порог вероятности для AI (default: 0.5)')
    args = parser.parse_args()

    # Загрузка модели
    model_path = Path(args.model)
    if not model_path.exists():
        # Ищем рядом со скриптом
        model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"Ошибка: модель не найдена: {args.model}")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    model        = bundle['model']
    feature_cols = bundle['feature_cols']
    print(f"Модель загружена: {model_path.name}\n")

    # Обрабатываем каждый файл
    results = []
    for pattern in args.files:
        paths = sorted(Path().glob(pattern)) if '*' in pattern else [Path(pattern)]
        for path in paths:
            if not path.exists():
                print(f"Файл не найден: {path}")
                continue

            text = path.read_text(encoding='utf-8', errors='replace').strip()
            if len(text) < 100:
                print(f"{path.name}: слишком короткий текст (< 100 символов)")
                continue

            pred, p_human, p_ai = predict_text(
                text, model, feature_cols, verbose=args.verbose)

            if pred is None:
                print(f"{path.name}: не удалось извлечь признаки")
                continue

            # Применяем порог
            label = 'AI' if p_ai >= args.threshold else 'ЧЕЛОВЕК'
            conf  = p_ai if p_ai >= args.threshold else p_human
            bar   = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))

            print(f"{'─'*55}")
            print(f"Файл:    {path.name}")
            print(f"Вердикт: {label}  (уверенность: {conf:.1%})")
            print(f"         [{bar}]")
            print(f"         P(Human)={p_human:.3f}  P(AI)={p_ai:.3f}")

            if args.verbose:
                print()

            results.append({'file': str(path), 'label': label,
                             'p_human': p_human, 'p_ai': p_ai})

    # Итог если несколько файлов
    if len(results) > 1:
        print(f"\n{'='*55}")
        ai_count = sum(1 for r in results if r['label'] == 'AI')
        print(f"Итого файлов:  {len(results)}")
        print(f"  AI:          {ai_count}")
        print(f"  Человек:     {len(results) - ai_count}")


if __name__ == '__main__':
    main()
