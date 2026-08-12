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
