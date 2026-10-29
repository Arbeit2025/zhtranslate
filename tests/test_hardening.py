"""Regression tests for the 2026-10-29 predictive-hardening cycles (Muse).

Each group is numbered after the cycle that introduced it. See
docs/mini_dev_log.md for the predicted failure, the fix and the
verification record of each cycle.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from translator import AccuracyFirstChineseTranslator


@pytest.fixture
def translator():
    t = AccuracyFirstChineseTranslator()
    t._load_stanza = lambda lang: False
    return t


# --- Cycle 1: number formatting differences must not false-HOLD ---

def test_numbers_fullwidth_are_canonicalized():
    assert AccuracyFirstChineseTranslator._numbers('２０２６年に１２．５％増えた') == ['2026', '12.5%']
    assert AccuracyFirstChineseTranslator._numbers('２０２６年') == AccuracyFirstChineseTranslator._numbers('2026年')


def test_numbers_thousands_and_trailing_zeros():
    assert AccuracyFirstChineseTranslator._numbers('合計10,000元と12.50元') == ['10000', '12.5']
    assert AccuracyFirstChineseTranslator._numbers('合計10000元と12.5元') == ['10000', '12.5']
    assert AccuracyFirstChineseTranslator._numbers('欧州式 1.234.567 件') == ['1234567']
    assert AccuracyFirstChineseTranslator._numbers('小数 1,5 点') == ['1.5']


def test_numbers_percent_spacing():
    assert AccuracyFirstChineseTranslator._numbers('12.5%増') == AccuracyFirstChineseTranslator._numbers('增长12.5 %')


def test_numbers_real_change_still_detected():
    assert AccuracyFirstChineseTranslator._numbers('12.5元') != AccuracyFirstChineseTranslator._numbers('12.6元')
    assert AccuracyFirstChineseTranslator._numbers('2026年') != AccuracyFirstChineseTranslator._numbers('2027年')


def test_reformatted_numbers_do_not_hold(translator):
    translator._translate_nllb = lambda text, *args: (
        '2026年的故事。' if '2026' in text else '价格是12.5美元，总计10000美元。', 10)
    result = translator.translate_document(
        '価格は12.50ドルで、合計10,000ドルです。2026年の話。',
        source='jpn_Jpan', run_backtranslation=False,
    )
    assert not any(
        c['code'] == 'NUMBER_PRESERVATION' and c['level'] == 'hold'
        for s in result['sentences'] for c in s['checks']
    )


# --- Cycle 2: paragraph breaks survive document translation ---

def test_paragraph_breaks_preserved(translator):
    result = translator.translate_document(
        '学习软件。\n\n网络运行。学习更多。',
        source='zho_Hans', target='tw', run_backtranslation=False,
    )
    assert result['translation'] == '學習軟體。\n\n網路執行。\n學習更多。'
    assert result['sentence_count'] == 3


def test_single_paragraph_unchanged(translator):
    result = translator.translate_document(
        '学习软件。网络运行。', source='zho_Hans', target='tw', run_backtranslation=False,
    )
    assert result['translation'] == '學習軟體。\n網路執行。'


def test_many_paragraphs_preserved(translator):
    text = '\n\n\n'.join(['学习软件。', '网络运行。', '学习软件。'])
    result = translator.translate_document(text, source='zho_Hans', target='cn', run_backtranslation=False)
    assert result['translation'].count('\n\n') == 2
    assert result['sentence_count'] == 3


# --- Cycle 3: glossary ASCII boundaries / TECH protection vs tokens ---

def test_glossary_ascii_key_not_applied_inside_word(translator):
    result = translator.translate_document(
        'AI技术与AIDS防治', source='zho_Hans', target='cn',
        glossary_text='AI=人工智能', run_backtranslation=False,
    )
    assert result['translation'] == '人工智能技术与AIDS防治'
    assert 'ZZ' not in result['translation']
    assert result['status'] != 'HOLD'


def test_glossary_cjk_key_still_matches_inside_compound(translator):
    result = translator.translate_document(
        '自主学习很重要', source='zho_Hans', target='cn',
        glossary_text='学习=研修', run_backtranslation=False,
    )
    assert result['translation'] == '自主研修很重要'


def test_glossary_ascii_key_applies_at_boundaries(translator):
    result = translator.translate_document(
        'OpenAIとAI、AI技術', source='zho_Hans', target='cn',
        glossary_text='AI=人工智能', run_backtranslation=False,
    )
    assert result['translation'].count('人工智能') == 2


def test_tech_protection_skips_spans_containing_tokens(translator):
    from translator import TECH_TOKEN_RE
    protected, mapping = translator._protect('ZZTERM0001ZZDS', 'TECH', [TECH_TOKEN_RE.pattern])
    assert protected == 'ZZTERM0001ZZDS' and mapping == {}


def test_changed_number_still_holds(translator):
    translator._translate_nllb = lambda *args: ('价格是12.6美元。', 10)
    result = translator.translate_document(
        '価格は12.5ドルです。', source='jpn_Jpan', run_backtranslation=False,
    )
    assert result['status'] == 'HOLD'
    assert any(
        c['code'] == 'NUMBER_PRESERVATION' and c['level'] == 'hold'
        for s in result['sentences'] for c in s['checks']
    )
