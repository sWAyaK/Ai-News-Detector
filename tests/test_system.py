"""Системные тесты — REST API, CLI, качество модели, производительность."""
import io
import os
import sys
import time
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_AI_TEXT = ("В современном мире искусственный интеллект играет "
                  "важную роль. Таким образом, можно сделать вывод. "
                  "Следует отметить значимость данной области. " * 5)


@pytest.fixture(scope='session')
def model_bundle():
    with open(PROJECT_ROOT / 'model_gb.pkl', 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='session')
def client():
    sys.path.insert(0, str(PROJECT_ROOT))
    import web_app
    web_app.app.config['TESTING'] = True
    with web_app.app.test_client() as c:
        yield c


# ── Качество модели ──────────────────────────────────────────────────
class TestModelQuality:

    def test_roc_auc(self, model_bundle, feature_matrix):
        X, y = feature_matrix
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
        proba = model_bundle['model'].predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        assert auc >= 0.90

    def test_f1(self, model_bundle, feature_matrix):
        X, y = feature_matrix
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
        proba = model_bundle['model'].predict_proba(X_te)[:, 1]
        best = max(f1_score(y_te, (proba >= t).astype(int), average='macro')
                   for t in np.arange(0.30, 0.71, 0.02))
        assert best >= 0.85


# ── REST API ─────────────────────────────────────────────────────────
class TestRestAPI:

    def test_analyze_ai_text(self, client):
        r = client.post('/api/analyze', json={'text': SAMPLE_AI_TEXT})
        assert r.status_code == 200
        body = r.get_json()
        assert 'p_ai' in body and 'p_human' in body

    def test_analyze_too_short(self, client):
        r = client.post('/api/analyze', json={'text': 'Привет'})
        assert 400 <= r.status_code < 500

    def test_news_endpoint(self, client):
        r = client.get('/api/news?src=lenta')
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_analyze_file_utf8(self, client):
        data = {'file': (io.BytesIO(SAMPLE_AI_TEXT.encode('utf-8')),
                          'test.txt')}
        r = client.post('/api/analyze_file', data=data,
                         content_type='multipart/form-data')
        assert r.status_code == 200


# ── Командный режим ─────────────────────────────────────────────────
class TestCLI:

    def test_single_file(self, tmp_path):
        f = tmp_path / 'sample.txt'
        f.write_text(SAMPLE_AI_TEXT, encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        r = subprocess.run([sys.executable, 'predict.py', str(f)],
                            cwd=PROJECT_ROOT,
                            capture_output=True, text=True,
                            encoding='utf-8', env=env, timeout=30)
        assert r.returncode == 0
        assert 'AI' in r.stdout or 'ЧЕЛОВЕК' in r.stdout

    def test_missing_file(self, tmp_path):
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        r = subprocess.run([sys.executable, 'predict.py',
                            str(tmp_path / 'no.txt')],
                            cwd=PROJECT_ROOT,
                            capture_output=True, text=True,
                            encoding='utf-8', env=env, timeout=30)
        out = (r.stdout + r.stderr).lower()
        assert 'не найден' in out or 'not found' in out


# ── Производительность ─────────────────────────────────────────────
class TestPerformance:

    def test_latency(self, client):
        client.post('/api/analyze', json={'text': SAMPLE_AI_TEXT})  # прогрев
        timings = []
        for _ in range(30):
            t0 = time.perf_counter()
            client.post('/api/analyze', json={'text': SAMPLE_AI_TEXT})
            timings.append((time.perf_counter() - t0) * 1000)
        timings.sort()
        assert timings[15] <= 5000

    def test_no_memory_leak(self, client):
        import psutil
        proc = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / (1024 * 1024)
        for _ in range(100):
            client.post('/api/analyze', json={'text': SAMPLE_AI_TEXT})
        mem_after = proc.memory_info().rss / (1024 * 1024)
        assert mem_after - mem_before < 50
