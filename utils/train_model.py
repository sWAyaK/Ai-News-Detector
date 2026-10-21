"""
Переобучение модели GradientBoosting на объединённом датасете.

Шаги:
  1. Загружает dataset_combined_v2.csv (подготовленный clean_datasets.py)
  2. Извлекает признаки через extract_features.py
  3. Обучает LightGBM с подбором порога
  4. Сохраняет model_gb.pkl в корне проекта

Запуск:
    python utils/train_model.py
    python utils/train_model.py --data datasets/dataset_combined_v2.csv --verbose
"""

import sys
import pickle
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    classification_report, confusion_matrix,
)

# ── Пути ──────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
BASE    = HERE.parent
sys.path.insert(0, str(BASE))

from extract_features import extract_features

DATA_PATH  = BASE / 'datasets' / 'dataset_combined_v2.csv'
MODEL_PATH = BASE / 'model_gb.pkl'


# ── Параметры модели ──────────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators    = 500,
    learning_rate   = 0.04,
    max_depth       = 6,
    num_leaves      = 40,
    min_child_samples = 20,
    subsample       = 0.8,
    subsample_freq  = 1,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 0.1,
    class_weight    = 'balanced',
    random_state    = 42,
    n_jobs          = -1,
    verbose         = -1,
)


# ── Извлечение признаков ──────────────────────────────────────────────────────

def build_feature_matrix(texts: list, labels: list, verbose: bool = True):
    """Извлекает признаки и возвращает DataFrame + список меток."""
    records = []
    skip    = 0
    t0      = time.time()

    for i, (text, lbl) in enumerate(zip(texts, labels)):
        feat = extract_features(str(text))
        if not feat:
            skip += 1
            continue
        feat['label'] = int(lbl)
        records.append(feat)

        if verbose and (i + 1) % 500 == 0:
            pct = (i + 1) / len(texts) * 100
            print(f'  {i+1}/{len(texts)} ({pct:.0f}%) — {time.time()-t0:.0f}с', flush=True)

    print(f'  Готово за {time.time()-t0:.0f}с. Пропущено (мало слов): {skip}')
    df = pd.DataFrame(records)
    return df


# ── Основной пайплайн ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Обучение модели')
    parser.add_argument('--data', default=str(DATA_PATH),
                        help='Путь к подготовленному датасету')
    parser.add_argument('--model', default=str(MODEL_PATH),
                        help='Путь для сохранения модели')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--cv', action='store_true',
                        help='Дополнительно провести 5-fold кросс-валидацию')
    args = parser.parse_args()

    # ── 1. Загрузка датасета ──────────────────────────────────────────────────
    print('=' * 60)
    print(f'ДАННЫЕ: {args.data}')
    print('=' * 60)
    df = pd.read_csv(args.data)
    df['text'] = df['text'].fillna('').str.strip()
    df = df[df['text'].str.len() >= 100]
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df[df['label'].isin([0, 1])]

    vc = df['label'].value_counts().sort_index()
    print(f'  Строк: {len(df)}  (human={vc.get(0,0)}, ai={vc.get(1,0)})')

    # ── 2. Извлечение признаков ───────────────────────────────────────────────
    print('\nИЗВЛЕЧЕНИЕ ПРИЗНАКОВ')
    print('=' * 60)
    feat_df = build_feature_matrix(
        df['text'].tolist(),
        df['label'].tolist(),
        verbose=args.verbose or len(df) > 1000,
    )
    print(f'  Признаков: {feat_df.shape[1] - 1}  Примеров: {len(feat_df)}')

    feature_cols = [c for c in feat_df.columns if c != 'label']
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = feat_df['label']

    # ── 3. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'\n  Train: {len(X_train)}  Test: {len(X_test)}')

    # ── 4. Обучение ───────────────────────────────────────────────────────────
    print('\nОБУЧЕНИЕ LightGBM')
    print('=' * 60)
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    # Early stopping через eval_set
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=50 if args.verbose else 0),
        ],
    )
    print(f'  Лучшая итерация: {model.best_iteration_}')

    # ── 5. Оценка ─────────────────────────────────────────────────────────────
    print('\nОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ')
    print('=' * 60)
    proba  = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, proba)
    print(f'  ROC AUC:  {auc:.4f}')

    # Подбор оптимального порога по F1
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.02):
        preds = (proba >= thr).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, preds, average='macro')
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_pred = (proba >= best_thr).astype(int)
    acc    = accuracy_score(y_test, y_pred)
    print(f'  Accuracy: {acc:.4f}  (порог={best_thr:.2f})')
    print('\nConfusion Matrix (строки=факт, столбцы=предсказание):')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred,
                                 target_names=['Human', 'AI'], digits=3))

    # ── Топ признаков ─────────────────────────────────────────────────────────
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print('Топ-15 признаков по важности:')
    for feat, val in imp.head(15).items():
        print(f'  {feat:<28} {val}')

    # ── 6. (Опционально) Кросс-валидация ─────────────────────────────────────
    if args.cv:
        print('\nКРОСС-ВАЛИДАЦИЯ (5-fold)')
        print('=' * 60)
        skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs   = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            m = lgb.LGBMClassifier(**LGBM_PARAMS)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(40, verbose=False),
                              lgb.log_evaluation(0)])
            p = m.predict_proba(X.iloc[va_idx])[:, 1]
            a = roc_auc_score(y.iloc[va_idx], p)
            aucs.append(a)
            print(f'  Fold {fold}: AUC={a:.4f}')
        print(f'  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')

    # ── 7. Сохранение ─────────────────────────────────────────────────────────
    out = Path(args.model)
    bundle = {
        'model':        model,
        'feature_cols': feature_cols,
        'threshold':    best_thr,
        'auc':          round(auc, 4),
    }
    with open(out, 'wb') as f:
        pickle.dump(bundle, f)

    print(f'\n  Модель сохранена: {out}')
    print(f'  ROC AUC={auc:.4f}  Accuracy={acc:.4f}  Порог={best_thr:.2f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
