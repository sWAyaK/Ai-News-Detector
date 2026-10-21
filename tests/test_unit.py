"""Модульные тесты функций модулей extract_features.py и web_app.py."""
import pytest
from app.extract_features import tokenize, sent_tokenize, extract_features

class TestTokenize:
    def test_extracts_cyrillic_only(self):
        result = tokenize("Это test 123 текст с латиницей.")
        assert result == ['это', 'текст', 'с', 'латиницей']

    def test_lowercases(self):
        assert tokenize("Москва Питер") == ['москва', 'питер']

    def test_empty(self):
        assert tokenize("") == []


class TestSentTokenize:
    def test_splits_on_terminals(self):
        sents = sent_tokenize("Первое предложение. Второе! Третье?")
        assert len(sents) == 3

    def test_skips_short(self):
        sents = sent_tokenize("А. Длинное предложение для теста.")
        assert len(sents) == 1


class TestExtractFeatures:
    def test_returns_full_schema(self):
        feat = extract_features("Это пробный русский текст. " * 30)
        assert feat is not None and len(feat) == 32

    def test_returns_none_for_short(self):
        assert extract_features("Привет") is None

    def test_ai_markers_detected(self):
        ai = ("В современном мире искусственный интеллект играет важную роль. "
              "Таким образом, можно сделать вывод. Следует отметить.")
        plain = ("Госдума приняла закон. Сенатор внёс поправки. "
                 "Третье чтение запланировано на 19 мая.")
        ai_feat = extract_features(ai * 2)
        plain_feat = extract_features(plain * 2)
        assert ai_feat['ai_marker_density'] > plain_feat['ai_marker_density']

    def test_ttr_in_range(self):
        feat = extract_features("Простой текст. " * 50)
        assert 0 <= feat['ttr'] <= 1

    def test_word_count_positive(self):
        feat = extract_features("Тестовый текст для проверки. " * 20)
        assert feat['word_count'] > 0


class TestCleanWebText:
    def test_removes_subscribe_block(self):
        from app.web_app import _clean_web_text
        dirty = "Содержание статьи. Подписывайтесь на наш телеграм-канал."
        clean = _clean_web_text(dirty)
        assert 'Подписывайтесь' not in clean

    def test_preserves_content(self):
        from app.web_app import _clean_web_text
        original = "Важный новостной текст без шаблонных хвостов."
        assert original in _clean_web_text(original)
