"""Tests for the Azure Translator hosted engine (2026-10-29).

The HTTP layer is faked via urllib.request.urlopen; no network or real
Azure key is involved. Local NLLB remains the default engine when no
Azure configuration is present.
"""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import translator as translator_module
from translator import AccuracyFirstChineseTranslator, EngineRequestError


@pytest.fixture
def azure_translator(monkeypatch):
    monkeypatch.setenv('AZURE_TRANSLATOR_KEY', 'test-key')
    monkeypatch.delenv('MT_ENGINE', raising=False)
    t = AccuracyFirstChineseTranslator()
    t._load_stanza = lambda lang: False
    return t


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode('utf-8')

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fake_urlopen_factory(payload, captured):
    def fake_urlopen(request, timeout=None):
        captured['url'] = request.full_url
        captured['headers'] = dict(request.header_items())
        captured['body'] = json.loads(request.data.decode('utf-8'))
        return FakeResponse(payload)
    return fake_urlopen


# --- engine selection ---

def test_engine_defaults_to_nllb_without_azure_env(monkeypatch):
    monkeypatch.delenv('AZURE_TRANSLATOR_KEY', raising=False)
    monkeypatch.delenv('MT_ENGINE', raising=False)
    t = AccuracyFirstChineseTranslator()
    assert t._engine() == 'nllb'
    assert t._engine_name() == translator_module.MODEL_NAME


def test_engine_azure_selected_by_key(monkeypatch):
    monkeypatch.setenv('AZURE_TRANSLATOR_KEY', 'k')
    monkeypatch.delenv('MT_ENGINE', raising=False)
    t = AccuracyFirstChineseTranslator()
    assert t._engine() == 'azure'
    assert t._engine_name() == 'azure-translator'
    assert t.engine_ready() is True
    assert t.model_state == 'api_ready'


def test_engine_explicit_mt_engine_wins(monkeypatch):
    monkeypatch.setenv('AZURE_TRANSLATOR_KEY', 'k')
    monkeypatch.setenv('MT_ENGINE', 'nllb')
    assert AccuracyFirstChineseTranslator()._engine() == 'nllb'
    monkeypatch.delenv('AZURE_TRANSLATOR_KEY')
    monkeypatch.setenv('MT_ENGINE', 'azure')
    t = AccuracyFirstChineseTranslator()
    assert t._engine() == 'azure'
    assert t.engine_ready() is False
    assert t.model_state == 'api_unconfigured'


def test_azure_engine_not_blocked_by_low_memory(azure_translator):
    azure_translator.memory_limit_mb = 512
    assert azure_translator._insufficient_model_memory() is False


# --- language mapping ---

def test_azure_language_mapping():
    m = translator_module._AZURE_LANG
    assert m['jpn_Jpan'] == 'ja'
    assert m['eng_Latn'] == 'en'
    assert m['zho_Hans'] == 'zh-Hans'
    assert m['zho_Hant'] == 'zh-Hant'
    assert m['kor_Hang'] == 'ko'
    assert m['zsm_Latn'] == 'ms'
    assert m['arb_Arab'] == 'ar'


# --- request / response handling ---

def test_azure_translate_request_and_response(azure_translator, monkeypatch):
    captured = {}
    payload = [{'translations': [{'text': '印度是南亚的国家。', 'to': 'zh-Hans'}]}]
    monkeypatch.setattr(translator_module.urllib.request, 'urlopen', fake_urlopen_factory(payload, captured))
    out, count = azure_translator._translate_azure('インドは南アジアにある国家である。', 'jpn_Jpan', 'zho_Hans')
    assert out == '印度是南亚的国家。'
    assert count == 0
    assert 'from=ja' in captured['url'] and 'to=zh-Hans' in captured['url']
    headers_ci = {k.lower(): v for k, v in captured['headers'].items()}
    assert headers_ci.get('ocp-apim-subscription-key') == 'test-key'
    assert captured['body'] == [{'text': 'インドは南アジアにある国家である。'}]


def test_azure_translate_traditional_target(azure_translator, monkeypatch):
    captured = {}
    payload = [{'translations': [{'text': '印度是南亞的國家。', 'to': 'zh-Hant'}]}]
    monkeypatch.setattr(translator_module.urllib.request, 'urlopen', fake_urlopen_factory(payload, captured))
    out, _ = azure_translator._translate_azure('テスト。', 'jpn_Jpan', 'zho_Hant')
    assert out == '印度是南亞的國家。'
    assert 'to=zh-Hant' in captured['url']


def test_azure_missing_key_raises(monkeypatch):
    monkeypatch.delenv('AZURE_TRANSLATOR_KEY', raising=False)
    monkeypatch.setenv('MT_ENGINE', 'azure')
    t = AccuracyFirstChineseTranslator()
    with pytest.raises(EngineRequestError):
        t._translate_azure('テスト。', 'jpn_Jpan', 'zho_Hans')


def test_azure_quota_error_becomes_user_facing(azure_translator, monkeypatch):
    def failing_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 403, 'Forbidden', {},
            io.BytesIO(json.dumps({'error': {'code': 403001, 'message': 'Out of call volume quota'}}).encode()))
    monkeypatch.setattr(translator_module.urllib.request, 'urlopen', failing_urlopen)
    with pytest.raises(EngineRequestError) as excinfo:
        azure_translator._translate_azure('テスト。', 'jpn_Jpan', 'zho_Hans')
    assert '無料枠' in str(excinfo.value)


def test_azure_malformed_response_raises(azure_translator, monkeypatch):
    captured = {}
    monkeypatch.setattr(translator_module.urllib.request, 'urlopen', fake_urlopen_factory([{'unexpected': True}], captured))
    with pytest.raises(EngineRequestError):
        azure_translator._translate_azure('テスト。', 'jpn_Jpan', 'zho_Hans')


# --- end to end through translate_document ---

def test_azure_end_to_end_document(azure_translator, monkeypatch):
    captured = {}
    payload = [{'translations': [{'text': '印度是南亚的国家。', 'to': 'zh-Hans'}]}]
    monkeypatch.setattr(translator_module.urllib.request, 'urlopen', fake_urlopen_factory(payload, captured))
    result = azure_translator.translate_document(
        'インドは南アジアにある国家である。', source='jpn_Jpan', run_backtranslation=False)
    assert result['engine'] == 'azure-translator'
    assert result['translation'] == '印度是南亚的国家。'
    assert not any(
        c['code'] == 'MODEL_INPUT' and c['level'] == 'hold'
        for s in result['sentences'] for c in s['checks'])
