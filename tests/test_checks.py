import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator import AccuracyFirstChineseTranslator, TARGETS

T = AccuracyFirstChineseTranslator()


def test_split_japanese():
    assert T.split_sentences('今日は晴れ。明日は雨。') == ['今日は晴れ。', '明日は雨。']


def test_split_chinese():
    assert T.split_sentences('今天晴天。明天下雨！') == ['今天晴天。', '明天下雨！']


def test_split_latin():
    assert T.split_sentences('This is one. This is two!') == ['This is one.', 'This is two!']


def test_glossary():
    g = T.parse_glossary('自己調整学習=自我调节学习\nAI\tAI')
    assert g['自己調整学習'] == '自我调节学习'
    assert g['AI'] == 'AI'


def test_number_extraction():
    assert T._numbers('2026年に12.5%増えた。') == ['2026', '12.5%']


def test_roundtrip_identity():
    assert T.roundtrip_similarity('abc 123', 'abc 123') == 100.0


def test_targets_exist():
    assert TARGETS['cn']['code'] == 'zho_Hans'
    assert TARGETS['tw']['code'] == 'zho_Hant'
    assert TARGETS['hk']['code'] == 'zho_Hant'


def test_technical_tokens():
    assert T._technical_tokens('OpenAI uses GPT4 and HTML5.') == ['OpenAI', 'GPT4', 'HTML5']


# --- Regression tests for the 2026-10-29 ChatGPT-run findings ---

def test_technical_tokens_cjk_adjacent():
    # \b does not fire between Latin letters and kana/kanji, so the
    # old pattern found nothing at all in this sentence.
    assert T._technical_tokens('OpenAIはGPT4とHTML5を使用する。') == ['OpenAI', 'GPT4', 'HTML5']
    assert T._technical_tokens('使用GPT4模型') == ['GPT4']


def _stub_translator(output):
    inst = AccuracyFirstChineseTranslator()
    inst._translate_nllb = lambda text, src, tgt: (output, 10)
    return inst


def test_tech_token_dropout_is_hold():
    inst = _stub_translator('使用GPT4和HTML5。')  # OpenAI dropped
    r = inst.translate_sentence('OpenAIはGPT4とHTML5を使用する。', 'jpn_Jpan', 'cn', 'formal', {}, False)
    assert r['status'] == 'HOLD'
    assert any(c['code'] == 'TECH_TOKEN' and c['level'] == 'hold' for c in r['checks'])


def test_split_url_query_not_broken():
    s = '詳細は https://example.com/search?q=hello&lang=ja を見てください。次です。'
    assert T.split_sentences(s) == [
        '詳細は https://example.com/search?q=hello&lang=ja を見てください。',
        '次です。',
    ]


def test_split_url_sentence_final_period():
    s = 'Check https://example.com/path?query=test. Next sentence.'
    assert T.split_sentences(s) == [
        'Check https://example.com/path?query=test.',
        'Next sentence.',
    ]


def test_url_dropout_is_hold():
    inst = _stub_translator('请看详情。')  # placeholder token dropped
    r = inst.translate_sentence(
        '詳細は https://example.com/search?q=hello を見てください。',
        'jpn_Jpan', 'cn', 'formal', {}, False,
    )
    assert r['status'] == 'HOLD'


def test_split_ref_internal_punctuation_not_broken():
    s = 'これはテストです<ref>出典は『何々』。第二文も含む。</ref>続きです。'
    assert T.split_sentences(s) == [s]


def test_ref_dropout_is_hold():
    inst = _stub_translator('这是测试。')  # placeholder token dropped
    r = inst.translate_sentence(
        'これはテストです<ref>出典は『何々』。第二文も含む。</ref>続きです。',
        'jpn_Jpan', 'cn', 'formal', {}, False,
    )
    assert r['status'] == 'HOLD'


def test_script_residuals():
    assert T._script_residuals('学习中文软件', 'tw') == sorted(set('学习软'))
    assert T._script_residuals('學習中文軟體', 'tw') == []
    assert T._script_residuals('學習中文軟體', 'cn') == sorted(set('學習軟體'))
    assert T._script_residuals('学习中文软件', 'cn') == []


def test_tw_normalization_converts_simplified():
    if T._opencc(TARGETS['tw']['opencc']) is None:
        return  # OpenCC unavailable: residual check still guards (below)
    converted, _checks = T._normalize_target_variant('学习中文软件网络', 'tw')
    assert converted == '學習中文軟體網路'
    assert T._script_residuals(converted, 'tw') == []


def test_hant_normalization_converts_simplified():
    if T._opencc(TARGETS['hant']['opencc']) is None:
        return
    converted, _checks = T._normalize_target_variant('学习中文', 'hant')
    assert converted == '學習中文'


def test_script_mismatch_is_hold_when_conversion_unavailable():
    inst = _stub_translator('学习中文软件')
    inst._opencc = lambda config: None  # simulate OpenCC failure
    r = inst.translate_sentence('中国語を学ぶ。', 'jpn_Jpan', 'tw', 'formal', {}, False)
    assert r['status'] == 'HOLD'
    assert any(c['code'] == 'SCRIPT_MISMATCH' and c['level'] == 'hold' for c in r['checks'])


def test_glossary_dropout_is_hold():
    inst = _stub_translator('这是测试。')  # glossary token dropped
    r = inst.translate_sentence(
        '人工知能を使う。', 'jpn_Jpan', 'cn', 'formal',
        {'人工知能': '人工智能'}, False,
    )
    assert r['status'] == 'HOLD'
