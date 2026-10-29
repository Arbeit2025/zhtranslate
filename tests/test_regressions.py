import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from translator import AccuracyFirstChineseTranslator, PROTECT_PATTERNS
import app as web


@pytest.fixture
def translator():
    t = AccuracyFirstChineseTranslator()
    t._load_stanza = lambda lang: False
    return t


def test_unicode_urls_nested_wiki_and_literal_tokens(translator):
    text = 'ZZPROT0000ZZ {{outer|x={{inner|體育}}}} [[體育]] https://example.com/體育?q=1&x=2'
    protected, mapping = translator._protect(text, 'PROT', PROTECT_PATTERNS)
    restored, missing = translator._restore(protected, mapping)
    assert restored == text and not missing
    assert '{{outer|x={{inner|體育}}}}' in mapping.values()
    assert 'https://example.com/體育?q=1&x=2' in mapping.values()


@pytest.mark.parametrize('target', ['cn', 'tw', 'hk', 'hant'])
def test_protected_script_is_unchanged(translator, target):
    text = '学习 [[體育]] {{outer|x={{inner|體育}}}} <ref>出典。補足。</ref> https://example.com/體育?q=1&x=2'
    result = translator.translate_document(text, source='zho_Hans', target=target, run_backtranslation=False)
    for literal in ['[[體育]]', '{{outer|x={{inner|體育}}}}', '<ref>出典。補足。</ref>', 'https://example.com/體育?q=1&x=2']:
        assert literal in result['translation']
    assert result['status'] != 'HOLD'


def test_translation_keeps_every_glossary_and_name_occurrence(translator):
    seen = []

    def model(text, src, tgt):
        seen.append(text)
        return '译文。', 4

    translator._translate_nllb = model
    result = translator.translate_document(
        '自己調整学習と自己調整学習、OpenAIとOpenAIの説明。',
        source='jpn_Jpan', glossary_text='自己調整学習=自我调节学习', run_backtranslation=False,
    )
    assert result['translation'].count('自我调节学习') == 2
    assert result['translation'].count('OpenAI') == 2
    assert result['status'] == 'REVIEW'
    assert seen and all('ZZ' not in text and 'OpenAI' not in text for text in seen)


def test_glossary_is_not_applied_inside_url_or_wiki(translator):
    result = translator.translate_document(
        '学习 https://example.com/学习 [[学习]] 学习', source='zho_Hans', target='tw',
        glossary_text='学习=研修\nZZ=破損\nPROT=破損', run_backtranslation=False,
    )
    assert result['translation'] == '研修 https://example.com/学习 [[学习]] 研修'
    assert result['status'] != 'HOLD'


def test_chinese_glossary_never_loads_nllb(translator):
    def forbidden(*args):
        pytest.fail('Chinese conversion must not load NLLB')
    translator._translate_nllb = forbidden
    result = translator.translate_document('学习软件', source='zho_Hans', target='tw', glossary_text='学习=学習', run_backtranslation=False)
    assert result['translation'] == '學習軟體'


def test_unit_numbers_are_extracted():
    assert AccuracyFirstChineseTranslator._numbers('10km 25kg -3.5m 2026年 GPT4 HTML5') == ['10', '25', '-3.5', '2026']


def test_english_prose_is_not_mistaken_for_technical_names(translator):
    from translator import TECH_TOKEN_RE
    text = 'The capital of China is Beijing.'
    prepared, mapping = translator._protect(text, 'TECH', [TECH_TOKEN_RE.pattern])
    assert prepared == text and mapping == {}


def test_empty_model_output_is_hold(translator):
    translator._translate_nllb = lambda *args: ('', 2)
    result = translator.translate_document('今日は晴れ。', source='jpn_Jpan', run_backtranslation=False)
    assert result['status'] == 'HOLD'
    assert any(c['code'] == 'EMPTY_TRANSLATION' for c in result['sentences'][0]['checks'])


def test_empty_prose_fragment_is_hold(translator):
    translator._translate_nllb = lambda *args: ('', 2)
    result = translator.translate_document('OpenAIについて。', source='jpn_Jpan', run_backtranslation=False)
    assert result['status'] == 'HOLD' and not result['translation']


def test_literal_placeholder_survives(translator):
    result = translator.translate_document('ZZPROT0000ZZ 学习', source='zho_Hans', run_backtranslation=False)
    assert result['translation'] == 'ZZPROT0000ZZ 学习'
    assert result['status'] != 'HOLD'


def test_restore_is_not_recursive(translator):
    text = 'ZZTERM0000ZZ 学习'
    result = translator.translate_document(text, source='zho_Hans', glossary_text='学习=研修', run_backtranslation=False)
    assert result['translation'] == 'ZZTERM0000ZZ 研修'
    assert result['status'] != 'HOLD'


@pytest.mark.parametrize('payload', [None, [], ['x'], 'x', 1, True, {'text': 1}, {'text': None}, {'text': 'x', 'backtranslation': 'false'}, {'text': 'x', 'glossary': {}}, {'text': 'x', 'target': []}])
def test_invalid_api_payloads_are_json_400(payload):
    with web.app.test_client() as client:
        response = client.post('/api/translate', json=payload)
    assert response.status_code == 400
    assert isinstance(response.get_json()['error'], str)


@pytest.mark.parametrize('overrides', [{'source': 'bad'}, {'target': 'bad'}, {'style': 'bad'}, {'text': ''}])
def test_api_options_are_validated(overrides):
    with web.app.test_client() as client:
        response = client.post('/api/translate', json={'text': '学习', 'source': 'zho_Hans', **overrides})
    assert response.status_code == 400


def test_model_unavailable_is_503_and_lock_is_released(monkeypatch):
    from translator import ModelUnavailableError
    def fail(**kwargs):
        raise ModelUnavailableError('test failure')
    monkeypatch.setattr(web.translator, 'translate_document', fail)
    with web.app.test_client() as client:
        response = client.post('/api/translate', json={'text': 'test', 'source': 'eng_Latn'})
    assert response.status_code == 503 and response.get_json()['code'] == 'MODEL_UNAVAILABLE'
    assert not web.translation_lock.locked()


def test_busy_does_not_start_second_translation():
    web.translation_lock.acquire()
    try:
        with web.app.test_client() as client:
            response = client.post('/api/translate', json={'text': 'test'})
            health = client.get('/api/health')
        assert response.status_code == 429 and response.headers['Retry-After'] == '5'
        assert health.get_json()['busy'] is True
    finally:
        web.translation_lock.release()


def test_health_does_not_claim_model_ready():
    t = web.translator
    with web.app.test_client() as client:
        response = client.get('/api/health')
    assert response.status_code == 200
    assert response.get_json()['translation_ready'] == t.model_loaded


def test_oversized_json_is_rejected():
    with web.app.test_client() as client:
        response = client.post('/api/translate', json={'text': 'x' * (256 * 1024)})
    assert response.status_code == 413 and response.is_json


def test_low_memory_conversion_does_not_import_torch(translator, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name in {'torch', 'stanza', 'transformers'}:
            pytest.fail('Heavy dependencies must not load for a low-memory conversion')
        return original_import(name, *args, **kwargs)

    t = AccuracyFirstChineseTranslator()
    t.memory_limit_mb = 512
    monkeypatch.setattr(builtins, '__import__', guard)
    result = t.translate_document('这个软件在网络上运行。', source='zho_Hans', target='tw', run_backtranslation=False)
    assert result['translation'] == '這個軟體在網路上執行。'
    assert result['status'] == 'REVIEW'


def test_low_memory_translation_is_actionable_503(monkeypatch):
    t = AccuracyFirstChineseTranslator()
    t.memory_limit_mb = 512
    monkeypatch.setattr(web, 'translator', t)
    with web.app.test_client() as client:
        response = client.post('/api/translate', json={'text': '今日は晴れ。', 'source': 'jpn_Jpan'})
    assert response.status_code == 503
    assert response.get_json()['code'] == 'INSUFFICIENT_MEMORY'
    assert '512MB' in response.get_json()['error']
    assert not t.model_loaded and not web.translation_lock.locked()


def test_gunicorn_config_keeps_health_responsive(monkeypatch):
    import runpy
    monkeypatch.setenv('PORT', '10000')
    cfg = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'gunicorn.conf.py'))
    assert cfg['bind'] == '0.0.0.0:10000'
    assert cfg['workers'] == 1 and cfg['threads'] >= 2 and cfg['timeout'] == 300
