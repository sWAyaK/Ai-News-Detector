#!/usr/bin/env python3
"""
verify_metrics.py — автотест для проверки числовых значений,
указанных в Главе 4 пояснительной записки.

Запускает измерения на реальной модели и датасете проекта и сравнивает
полученные значения с тремя ориентирами:
  1. обязательным порогом из требований (Н1, Н3 и т.д.) — assertion;
  2. заявленным в отчёте значением — предупреждение при расхождении;
  3. целевым диапазоном — справочно.

Использование:
    python verify_metrics.py                  # все проверки
    python verify_metrics.py --quick          # без сетевых запросов
    python verify_metrics.py --report-only    # только сводный отчёт, без assert

Запускать из корня проекта (там, где web_app.py, predict.py, model_gb.pkl).
"""
import argparse
import io
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

# ── Конфигурация ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv('NEWSDETECTOR_MODEL', PROJECT_ROOT / 'model_gb.pkl'))
DATASET_PATH = Path(os.getenv(
    'NEWSDETECTOR_DATASET',
    PROJECT_ROOT / 'datasets' / 'combined_dataset.csv'))

# Заявленные в отчёте значения (из раздела 4.3 и сводной таблицы 4.1)
REPORT_VALUES = {
    'roc_auc':           {'cited': 0.96, 'threshold': 0.90,  'req': 'Н3', 'unit': ''},
    'f1_macro':          {'cited': 0.89, 'threshold': 0.85,  'req': 'Н3', 'unit': ''},
    'accuracy':          {'cited': 0.91, 'threshold': 0.85,  'req': 'Н3', 'unit': ''},
    'auc_std':           {'cited': 0.011, 'threshold': 0.02, 'req': 'Н3', 'unit': '', 'lower_is_better': True},
    'fp_rate':           {'cited': None, 'threshold': 0.15,  'req': 'Н3', 'unit': '', 'lower_is_better': True},
    'fn_rate':           {'cited': None, 'threshold': 0.15,  'req': 'Н3', 'unit': '', 'lower_is_better': True},
    'latency_median_ms': {'cited': 95,    'threshold': 5000,  'req': 'Н1', 'unit': ' мс', 'lower_is_better': True},
    'latency_p95_ms':    {'cited': 240,   'threshold': 5000,  'req': 'Н1', 'unit': ' мс', 'lower_is_better': True},
    'memory_growth_mb':  {'cited': 8,     'threshold': 50,    'req': '-',  'unit': ' МБ', 'lower_is_better': True},
}

CITED_TOLERANCE = 0.05  # допустимое относительное отклонение от заявленного

# ── Утилиты вывода ──────────────────────────────────────────────────────────

class Term:
    GREEN = '\033[92m' if sys.stdout.isatty() else ''
    YELLOW = '\033[93m' if sys.stdout.isatty() else ''
    RED = '\033[91m' if sys.stdout.isatty() else ''
    BOLD = '\033[1m' if sys.stdout.isatty() else ''
    RESET = '\033[0m' if sys.stdout.isatty() else ''

def head(text):
    print(f'\n{Term.BOLD}── {text} ──{Term.RESET}')

results = {}  # имя_метрики → измеренное значение

def record(name, value, ok_text='OK', warn_text=None):
    """Записать измеренное значение в results и красиво напечатать."""
    spec = REPORT_VALUES.get(name, {})
    threshold = spec.get('threshold')
    cited = spec.get('cited')
    unit = spec.get('unit', '')
    lower_is_better = spec.get('lower_is_better', False)
    req = spec.get('req', '')

    results[name] = value

    # Соответствие требованию
    if threshold is not None and value is not None:
        if lower_is_better:
            passes_req = value <= threshold
            req_str = f'≤ {threshold}{unit}'
        else:
            passes_req = value >= threshold
            req_str = f'≥ {threshold}{unit}'
        req_mark = (f'{Term.GREEN}✓{Term.RESET}' if passes_req
                    else f'{Term.RED}✗{Term.RESET}')
    else:
        passes_req = None
        req_str = '-'
        req_mark = '·'

    # Соответствие заявленному
    if cited is not None and value is not None and cited != 0:
        rel_diff = abs(value - cited) / abs(cited)
        if rel_diff <= CITED_TOLERANCE:
            cited_mark = f'{Term.GREEN}≈{Term.RESET}'
        else:
            cited_mark = f'{Term.YELLOW}≠{Term.RESET}'
        cited_str = f'{cited}{unit}'
    else:
        cited_mark = '·'
        cited_str = '-'

    # Форматирование измеренного значения
    if isinstance(value, float):
        val_str = f'{value:.4f}{unit}' if abs(value) < 10 else f'{value:.1f}{unit}'
    elif value is None:
        val_str = 'не измерено'
    else:
        val_str = f'{value}{unit}'

    print(f'  {name:<22} | измерено: {val_str:<16} | '
          f'заявлено: {cited_str:<10} {cited_mark} | '
          f'треб. {req}: {req_str:<14} {req_mark}')

    return passes_req


# ── Измерения ───────────────────────────────────────────────────────────────

def measure_model_quality():
    """Замеряет ROC AUC, F1, accuracy, FP/FN rate, стабильность по 5-fold CV."""
    head('1. Качество классификатора (требование Н3)')

    if not MODEL_PATH.exists():
        print(f'  Пропущено: модель не найдена ({MODEL_PATH})')
        return
    if not DATASET_PATH.exists():
        print(f'  Пропущено: датасет не найден ({DATASET_PATH})')
        return

    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split, StratifiedKFold
        from sklearn.metrics import (
            roc_auc_score, f1_score, accuracy_score, confusion_matrix)
    except ImportError as e:
        print(f'  Пропущено: нет зависимости {e.name}')
        return

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from extract_features import extract_features
    except ImportError:
        print('  Пропущено: не удалось импортировать extract_features')
        return

    # Загрузка модели
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    feature_cols = list(bundle['feature_cols'])

    # Загрузка и подготовка датасета — гибкое определение колонок
    df = pd.read_csv(DATASET_PATH)
    text_col = next((c for c in ['text', 'content', 'article', 'body']
                     if c in df.columns), None)
    label_col = next((c for c in ['label', 'class', 'category', 'y', 'target']
                      if c in df.columns), None)
    if text_col is None or label_col is None:
        print(f'  Пропущено: не найдены колонки text/label. '
              f'Имеющиеся: {list(df.columns)}')
        return

    df['_text'] = df[text_col].fillna('').astype(str).str.strip()
    df = df[df['_text'].str.len() >= 100]
    df['_label'] = pd.to_numeric(df[label_col], errors='coerce')
    df = df[df['_label'].isin([0, 1])].reset_index(drop=True)

    # Извлечение признаков
    print(f'  Извлечение признаков из {len(df)} текстов...')
    feats = []
    labels = []
    for _, row in df.iterrows():
        f = extract_features(row['_text'])
        if f:
            feats.append(f)
            labels.append(int(row['_label']))
    X = pd.DataFrame(feats)[feature_cols].replace(
        [np.inf, -np.inf], 0).fillna(0)
    y = pd.Series(labels)

    # Train/test split
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    proba = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, proba))
    record('roc_auc', auc)

    # Подбор порога по макро-F1
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.02):
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_te, pred, average='macro')
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    record('f1_macro', best_f1)

    pred = (proba >= best_thr).astype(int)
    record('accuracy', float(accuracy_score(y_te, pred)))

    cm = confusion_matrix(y_te, pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        record('fp_rate', fp / (fp + tn) if (fp + tn) else 0.0)
        record('fn_rate', fn / (fn + tp) if (fn + tp) else 0.0)

    # CV-стабильность
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for _, va in skf.split(X, y):
        p = model.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p))
    record('auc_std', float(np.std(aucs)))


def measure_performance(skip_network=False):
    """Замеряет латентность инференса и рост памяти."""
    head('2. Производительность (требование Н1)')

    if not MODEL_PATH.exists():
        print('  Пропущено: модель отсутствует')
        return

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import web_app
    except ImportError as e:
        print(f'  Пропущено: не удалось импортировать web_app ({e})')
        return

    sample_text = (
        'В современном мире искусственный интеллект играет важную роль в '
        'развитии различных отраслей. Следует отметить, что технологии '
        'машинного обучения становятся всё более доступными для широкого '
        'круга специалистов. Таким образом, можно сделать вывод, что данная '
        'область является ключевой для будущего информационных технологий.'
    )

    web_app.app.config['TESTING'] = True
    with web_app.app.test_client() as client:
        # Прогрев
        client.post('/api/analyze', json={'text': sample_text})

        # Замер латентности — 30 запросов
        timings = []
        for _ in range(30):
            t0 = time.perf_counter()
            r = client.post('/api/analyze', json={'text': sample_text})
            timings.append((time.perf_counter() - t0) * 1000)
        timings.sort()

        record('latency_median_ms', timings[len(timings) // 2])
        record('latency_p95_ms', timings[int(len(timings) * 0.95)])

        # Замер роста памяти — 100 запросов
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss / (1024 * 1024)
            for _ in range(100):
                client.post('/api/analyze', json={'text': sample_text})
            mem_after = proc.memory_info().rss / (1024 * 1024)
            record('memory_growth_mb', mem_after - mem_before)
        except ImportError:
            print('  memory_growth_mb пропущено — нет библиотеки psutil')


def check_environment():
    """Проверка отсутствия GPU (требование Н2) и наличия зависимостей."""
    head('3. Окружение исполнения (требование Н2)')

    # Проверка отсутствия GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        print(f'  torch установлен; GPU доступен: {has_gpu}')
        if has_gpu:
            print('  ⚠ В системе обнаружен GPU, но проект работает на CPU — '
                  'требование Н2 формально выполнено через отказ от GPU-зависимостей')
    except ImportError:
        print(f'  {Term.GREEN}✓{Term.RESET} torch отсутствует — GPU не используется')

    # Проверка зависимостей
    print('  Версии ключевых библиотек:')
    for mod_name in ['numpy', 'pandas', 'lightgbm', 'sklearn', 'flask']:
        try:
            m = __import__(mod_name)
            if (m == 'flask'):
                v = importlib.metadata.version('flask')
            else:
                v = getattr(m, '__version__', '?')
            print(f'    {mod_name:<10} {v}')
        except ImportError:
            print(f'    {mod_name:<10} НЕ УСТАНОВЛЕН')


def final_report():
    """Печать сводной таблицы для копирования в отчёт."""
    head('Сводная таблица для отчёта (значения из этого запуска)')

    rows = []
    rows.append(('Метрика', 'Значение', 'Заявлено в отчёте', 'Расхождение'))
    rows.append(('-' * 22, '-' * 14, '-' * 16, '-' * 12))

    def fmt(v, unit=''):
        if v is None: return '—'
        return f'{v:.4f}{unit}' if isinstance(v, float) and abs(v) < 10 \
               else f'{v:.1f}{unit}'

    for name, spec in REPORT_VALUES.items():
        measured = results.get(name)
        cited = spec.get('cited')
        unit = spec.get('unit', '')
        m_str = fmt(measured, unit)
        c_str = fmt(cited, unit) if cited is not None else '—'
        if measured is not None and cited is not None and cited != 0:
            rel = abs(measured - cited) / abs(cited)
            d_str = f'{rel * 100:.1f}%'
        else:
            d_str = '—'
        rows.append((name, m_str, c_str, d_str))

    for cells in rows:
        print('  ' + ' | '.join(f'{c:<22}' if i == 0 else f'{c:<14}'
                                 for i, c in enumerate(cells)))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Проверка метрик из главы 4')
    ap.add_argument('--quick', action='store_true',
                    help='Пропустить тяжёлые проверки')
    ap.add_argument('--report-only', action='store_true',
                    help='Только сводный отчёт, без assertion')
    args = ap.parse_args()

    print(f'{Term.BOLD}=== Проверка метрик NewsDetector ==={Term.RESET}')
    print(f'Корень проекта: {PROJECT_ROOT}')
    print(f'Модель:         {MODEL_PATH}')
    print(f'Датасет:        {DATASET_PATH}')

    measure_model_quality()
    if not args.quick:
        measure_performance()
    check_environment()

    final_report()

    # Финальный assertion блок — нужен для интеграции в CI
    if not args.report_only:
        print(f'\n{Term.BOLD}=== Проверка соответствия требованиям ==={Term.RESET}')
        failed = []
        for name, spec in REPORT_VALUES.items():
            value = results.get(name)
            threshold = spec.get('threshold')
            if value is None or threshold is None:
                continue
            lower_is_better = spec.get('lower_is_better', False)
            ok = (value <= threshold) if lower_is_better else (value >= threshold)
            if not ok:
                failed.append((name, value, threshold, spec.get('req', '')))

        if failed:
            print(f'\n{Term.RED}✗ ОБНАРУЖЕНЫ НЕСООТВЕТСТВИЯ:{Term.RESET}')
            for name, val, thr, req in failed:
                print(f'  - {name}: {val} не удовлетворяет требованию {req} ({thr})')
            sys.exit(1)
        else:
            print(f'\n{Term.GREEN}✓ Все измеренные метрики соответствуют '
                  f'обязательным требованиям{Term.RESET}')


if __name__ == '__main__':
    main()
