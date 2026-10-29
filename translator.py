from __future__ import annotations

import html
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple
from pathlib import Path


MODEL_NAME = os.getenv('MT_MODEL', 'facebook/nllb-200-distilled-600M')

# Output modes. NLLB chooses the base script; OpenCC then normalizes regional forms.
TARGETS = {
    'cn': {
        'code': 'zho_Hans',
        'label': '简体中文（中国大陆）',
        'stanza': 'zh-hans',
        'opencc': 't2s.json',
    },
    'tw': {
        'code': 'zho_Hant',
        'label': '繁體中文（台灣正體）',
        'stanza': 'zh-hant',
        # t2tw.json assumes Traditional input and leaves Simplified
        # characters untouched. s2twp.json converts Simplified leakage
        # to Taiwan Traditional (incl. Taiwan vocabulary) and is
        # idempotent on already-Traditional input.
        'opencc': 's2twp.json',
    },
    'hk': {
        'code': 'zho_Hant',
        'label': '繁體中文（香港）',
        'stanza': 'zh-hant',
        # Same reason as tw: t2hk.json cannot fix Simplified leakage.
        'opencc': 's2hk.json',
    },
    'hant': {
        'code': 'zho_Hant',
        'label': '繁體中文（通用）',
        'stanza': 'zh-hant',
        # Was None (no normalization at all), so Simplified leakage
        # from NLLB passed through unchanged.
        'opencc': 's2t.json',
    },
}

# langid ISO-639-1 -> NLLB/FLORES code
LANGS: Dict[str, Tuple[str, str]] = {
    'ja': ('jpn_Jpan', '日本語'),
    'en': ('eng_Latn', 'English'),
    'zh': ('zho_Hans', '中文'),
    'ko': ('kor_Hang', '한국어'),
    'de': ('deu_Latn', 'Deutsch'),
    'fr': ('fra_Latn', 'Français'),
    'es': ('spa_Latn', 'Español'),
    'pt': ('por_Latn', 'Português'),
    'it': ('ita_Latn', 'Italiano'),
    'nl': ('nld_Latn', 'Nederlands'),
    'pl': ('pol_Latn', 'Polski'),
    'ru': ('rus_Cyrl', 'Русский'),
    'uk': ('ukr_Cyrl', 'Українська'),
    'tr': ('tur_Latn', 'Türkçe'),
    'ar': ('arb_Arab', 'العربية'),
    'hi': ('hin_Deva', 'हिन्दी'),
    'th': ('tha_Thai', 'ไทย'),
    'vi': ('vie_Latn', 'Tiếng Việt'),
    'ms': ('zsm_Latn', 'Bahasa Melayu'),
    'id': ('ind_Latn', 'Bahasa Indonesia'),
}

# Manual Chinese source choices are separate because langid returns only zh.
MANUAL_CHINESE_SOURCES = [
    {'value': 'zho_Hans', 'label': '中文（简体字）'},
    {'value': 'zho_Hant', 'label': '中文（繁體字）'},
]
NLLB_TO_LABEL = {v[0]: v[1] for v in LANGS.values()}
NLLB_TO_LABEL['zho_Hans'] = '中文（简体字）'
NLLB_TO_LABEL['zho_Hant'] = '中文（繁體字）'

FORMAL_WARN_WORDS = {
    '啥', '咋', '咋样', '咋樣', '俺', '甭', '老铁', '老鐵', '给力', '給力',
    '牛逼', '卧槽', '臥槽', '贼', '賊', '蛮', '蠻', '超赞', '超讚',
}

PROTECT_PATTERNS = [
    # Preserve Unicode paths and query strings; stop at surrounding CJK
    # punctuation, whitespace, markup, or quotes.
    r'https?://[^\s<>\{\}"\'，。！？、；：]+',
    r'<ref\b[^>]*>[\s\S]*?</ref>',
    r'<ref\b[^>]*/>',
    r'</?ref\b[^>]*>',
    r'\{\{[^{}]{0,2000}\}\}',
    r'\[\[[^\[\]]{0,2000}\]\]',
    r'\[[0-9]+\]',
    r'ZZ(?:PROT|TERM|TECH)\d+ZZ',
]

NUMBER_RE = re.compile(r'(?<![0-9A-Za-z])[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?(?:\s*%)?(?!\d)')
# Japanese text often mixes full-width digits / punctuation (２０２６年,
# １２．５％). \d matches the digits, but ． and ％ do not match [.,%], so a
# full-width number was extracted in pieces and then failed the preservation
# comparison against a half-width translation. Normalize a copy of the text
# for extraction only.
_NUMBER_NORMALIZE = str.maketrans('０１２３４５６７８９．，％＋－', '0123456789.,%+-')
TOKEN_RE = re.compile(r'ZZ(?:PROT|TERM|TECH)\d+ZZ')
HAN_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
# Do NOT use \b here: in Python re, CJK kana/kanji are \w, so there is
# no word boundary between e.g. "OpenAI" and a following "は". With \b
# the pattern matched nothing in "OpenAIはGPT4とHTML5を使用する。"
# and the preservation check silently never ran. ASCII-only
# lookarounds treat any non-ASCII character as a boundary instead.
TECH_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Z0-9.-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+\d+[A-Za-z0-9.-]*|\d+[A-Za-z][A-Za-z0-9.-]*)(?![A-Za-z0-9])')

# Characters that exist only in one script (unambiguous pairs only;
# shared / ambiguous forms such as 台, 干, 发's counterpart readings
# are deliberately excluded to avoid false positives). Used as an
# OpenCC-independent safety net: if the wrong script survives
# normalization, the sentence is HOLD, never READY.
SIMPLIFIED_ONLY_CHARS = frozenset(
    '国学学习汉语测试简体个们为这对时间过还进远运长门问关开闭现点电车马龙风云热气压产业业务经经济历乐书画写读说话认记视声响图区医员场园县镇户务动劳团战争军'
    '软网络数应术据库脑头湾转达标备设计机统选页内显错确译论证评议题导师级'
    '习题录频药响'
)
TRADITIONAL_ONLY_CHARS = frozenset(
    '國學學習漢語測試簡體個們為這對時間過還進遠運長門問關開閉現點電車馬龍風雲熱氣壓產業務經經濟歷樂書畫寫讀說話認記視聲響圖區醫員場園縣鎮戶務動勞團戰爭軍'
    '軟網絡數應術據庫腦頭灣轉達標備設計機統選頁內顯錯確譯論證評議題導師級'
    '習題錄頻藥響'
)


@dataclass
class Check:
    level: str  # ok, review, hold
    code: str
    message: str


class ModelUnavailableError(RuntimeError):
    """Model setup failed; callers can report an actionable HTTP 503."""


class InsufficientMemoryError(ModelUnavailableError):
    """Known runtime limit cannot hold the default NLLB model."""


def memory_limit_mb() -> int | None:
    limits = []
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = Path(path).read_text().strip()
            if value != 'max' and 0 < int(value) < 2 ** 60:
                limits.append(int(value))
        except (OSError, ValueError):
            pass
    return min(limits) // (1024 * 1024) if limits else None


class AccuracyFirstChineseTranslator:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.model_state = 'not_loaded'
        self.memory_limit_mb = memory_limit_mb()
        if self._insufficient_model_memory():
            self.model_state = 'insufficient_memory'
        self.stanza_pipelines: Dict[str, object] = {}
        self.opencc_converters: Dict[str, object] = {}

    def _insufficient_model_memory(self) -> bool:
        return (MODEL_NAME == 'facebook/nllb-200-distilled-600M'
                and self.memory_limit_mb is not None and self.memory_limit_mb < 3072)

    @staticmethod
    def language_choices() -> List[dict]:
        rows = [{'value': 'auto', 'label': '自動判定'}]
        seen = set()
        for iso, (code, label) in LANGS.items():
            if iso == 'zh':
                continue
            rows.append({'value': code, 'label': label})
            seen.add(code)
        for row in MANUAL_CHINESE_SOURCES:
            if row['value'] not in seen:
                rows.append(row)
        return rows

    @staticmethod
    def target_choices() -> List[dict]:
        return [
            {'value': key, 'label': cfg['label']}
            for key, cfg in TARGETS.items()
        ]

    def _load_model(self) -> None:
        if self.model_loaded:
            return
        if self._insufficient_model_memory():
            self.model_state = 'insufficient_memory'
            raise InsufficientMemoryError(
                f'公開先のメモリが不足しています（上限 {self.memory_limit_mb}MB）。'
                'NLLBによる翻訳にはメモリに余裕のあるサーバーかローカル環境が必要です。'
                '中国語同士の表記変換はこのまま利用できます。'
            )
        self.model_state = 'loading'
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            if torch.cuda.is_available():
                model = model.to('cuda')
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                model = model.to('mps')
            model.eval()
            self.tokenizer, self.model = tokenizer, model
            self.model_loaded = True
            self.model_state = 'ready'
        except Exception as exc:
            self.model_state = 'error'
            self.tokenizer, self.model = None, None
            raise ModelUnavailableError('NLLB model setup failed') from exc

    def _load_stanza(self, lang: str) -> bool:
        if lang in self.stanza_pipelines:
            return True
        if os.getenv('DISABLE_STANZA') == '1':
            return False
        if self.memory_limit_mb is not None and self.memory_limit_mb < 2048:
            # Loading torch plus both parsers on the free 512MB runtime can
            # kill even a simple Chinese-to-Chinese conversion request.
            return False
        try:
            import stanza
            import torch
            try:
                pipe = stanza.Pipeline(
                    lang,
                    processors='tokenize,pos,lemma,depparse',
                    use_gpu=torch.cuda.is_available(),
                    verbose=False,
                    download_method=None,
                )
            except Exception:
                if os.getenv('AUTO_DOWNLOAD_STANZA', '0') != '1':
                    return False
                stanza.download(lang, verbose=False)
                pipe = stanza.Pipeline(
                    lang,
                    processors='tokenize,pos,lemma,depparse',
                    use_gpu=torch.cuda.is_available(),
                    verbose=False,
                    download_method=None,
                )
            self.stanza_pipelines[lang] = pipe
            return True
        except Exception:
            return False

    def _opencc(self, config: str):
        if config in self.opencc_converters:
            return self.opencc_converters[config]
        try:
            import opencc
            converter = opencc.OpenCC(config)
            self.opencc_converters[config] = converter
            return converter
        except Exception:
            return None

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        # Protect URLs / <ref> / templates / wikilinks BEFORE splitting.
        # Splitting first broke URLs at their '?' (query was severed)
        # and broke <ref> at sentence punctuation inside the tag, so
        # the fragments were never protected and could be dropped by
        # the model without ever triggering the placeholder check.
        protected_text, mapping = AccuracyFirstChineseTranslator._protect(
            text, 'PROT', PROTECT_PATTERNS
        )

        def restore(segment: str) -> str:
            for key, original in mapping.items():
                if key in segment:
                    segment = segment.replace(key, original)
            return segment

        out: List[str] = []
        for paragraph in re.split(r'\n{2,}', protected_text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # Split Japanese/Chinese punctuation even with no following space.
            # Latin punctuation splits when followed by whitespace.
            parts = re.split(r'(?<=[。！？!?])|(?<=[.!?])\s+', paragraph)
            for p in parts:
                p = p.strip()
                if p:
                    out.append(restore(p))
        return out or [text]

    @staticmethod
    def _protect(text: str, prefix: str, patterns: Iterable[str]) -> Tuple[str, Dict[str, str]]:
        mapping: Dict[str, str] = {}
        combined = re.compile('|'.join(f'(?:{p})' for p in patterns), re.I if prefix == 'PROT' else 0)
        matches = [(m.start(), m.end()) for m in combined.finditer(text)
                   if prefix != 'TECH' or not TOKEN_RE.fullmatch(m.group())]
        # Regex alone cannot protect nested templates/wikilinks. Add balanced
        # outer spans before choosing non-overlapping ranges.
        if prefix == 'PROT':
            for opening, closing in (('{{', '}}'), ('[[', ']]')):
                stack = []
                for m in re.finditer(re.escape(opening) + '|' + re.escape(closing), text):
                    if m.group() == opening:
                        stack.append(m.start())
                    elif stack:
                        start = stack.pop()
                        if not stack:
                            matches.append((start, m.end()))
        spans = []
        for start, end in sorted(matches, key=lambda x: (x[0], -x[1])):
            if not spans or start >= spans[-1][1]:
                spans.append((start, end))

        def repl(original: str) -> str:
            suffix = ''
            # A sentence-final '.' / '!' / '?' etc. after a URL is
            # punctuation, not part of the URL. Keep it outside the
            # protected span so sentence splitting still works there
            # and the URL itself is restored byte-for-byte.
            if original.startswith(('http://', 'https://')):
                core = original
                while core and core[-1] in '.,;:!?':
                    suffix = core[-1] + suffix
                    core = core[:-1]
                if core.endswith(')') and '(' not in core:
                    suffix = ')' + suffix
                    core = core[:-1]
                original = core
            index = len(mapping)
            key = f'ZZ{prefix}{index:04d}ZZ'
            while key in text or key in mapping:
                index += 1
                key = f'ZZ{prefix}{index:04d}ZZ'
            mapping[key] = original
            return key + suffix

        chunks, previous = [], 0
        for start, end in spans:
            chunks.extend((text[previous:start], repl(text[start:end])))
            previous = end
        chunks.append(text[previous:])
        return ''.join(chunks), mapping

    @staticmethod
    def parse_glossary(text: str) -> Dict[str, str]:
        glossary: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '\t' in line:
                src, tgt = line.split('\t', 1)
            elif '=' in line:
                src, tgt = line.split('=', 1)
            else:
                continue
            src, tgt = src.strip(), tgt.strip()
            if src and tgt:
                glossary[src] = tgt
        return glossary

    @staticmethod
    def _protect_glossary(text: str, glossary: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        mapping: Dict[str, str] = {}
        if not glossary:
            return text, mapping
        terms = re.compile('|'.join(re.escape(src) for src in sorted(glossary, key=len, reverse=True)))

        def replace(match):
            index = len(mapping)
            key = f'ZZTERM{index:04d}ZZ'
            while key in text or key in mapping:
                index += 1
                key = f'ZZTERM{index:04d}ZZ'
            mapping[key] = glossary[match.group()]
            return key

        # Protect every occurrence separately, and never match inside a PROT
        # token (e.g. a user glossary entry named "ZZ" or "PROT").
        parts = TOKEN_RE.split(text)
        tokens = TOKEN_RE.findall(text)
        out = []
        for i, part in enumerate(parts):
            out.append(terms.sub(replace, part))
            if i < len(tokens):
                out.append(tokens[i])
        return ''.join(out), mapping

    @staticmethod
    def _restore(text: str, mapping: Dict[str, str]) -> Tuple[str, List[str]]:
        missing = [key for key in mapping if text.count(key) != 1]
        if not mapping:
            return text, missing
        # One pass: restored literal text may itself resemble another key.
        pattern = re.compile('|'.join(re.escape(key) for key in mapping))
        return pattern.sub(lambda match: mapping[match.group()], text), missing

    def _translate_preserving(self, text: str, mapping: Dict[str, str], src: str, tgt: str) -> Tuple[str, int]:
        """Translate prose spans; keep protected material out of model generation.

        NLLB has no guarantee of copying arbitrary placeholder strings. Joining
        independently translated spans preserves each URL/term exactly once.
        This can reduce fluency, so the caller adds a REVIEW for split prose.
        """
        if not mapping:
            return self._translate_nllb(text, src, tgt)
        pattern = re.compile('(' + '|'.join(re.escape(k) for k in mapping) + ')')
        output, token_count = [], 0
        for part in pattern.split(text):
            if not part:
                continue
            if part in mapping or not re.search(r'[^\W\d_]|\d', part, re.UNICODE):
                output.append(part)
                continue
            translated, count = self._translate_nllb(part.strip(), src, tgt)
            if not translated.strip():
                raise ValueError('翻訳モデルが本文の一部に空の訳文を返しました。')
            token_count += count
            if token_count > 500:
                raise ValueError('単文が500 NLLB tokensを超えています。文を分けてください。')
            leading = part[:len(part) - len(part.lstrip())]
            trailing = part[len(part.rstrip()):]
            output.append(leading + translated + trailing)
        return ''.join(output), token_count

    @staticmethod
    def _normalize_for_similarity(text: str) -> str:
        text = html.unescape(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u3000\s]+', ' ', text)
        return text

    @classmethod
    def roundtrip_similarity(cls, source: str, back: str) -> float:
        a = cls._normalize_for_similarity(source)
        b = cls._normalize_for_similarity(back)
        if not a or not b:
            return 0.0
        return round(100.0 * SequenceMatcher(None, a, b).ratio(), 1)

    @staticmethod
    def _canonical_number(raw: str) -> str:
        """Canonical form for comparison: 10,000 / 10000, 12.50 / 12.5 and
        12.5% / 12.5 % are the same number; 12.5 vs 12.6 still differ."""
        from decimal import Decimal, InvalidOperation
        percent = raw.rstrip().endswith('%')
        s = re.sub(r'\s+', '', raw).rstrip('%')
        try:
            if re.fullmatch(r'[+-]?\d{1,3}([.,]\d{3})+', s):
                s = re.sub(r'[.,]', '', s)  # pure thousands grouping
            elif ',' in s and '.' in s:
                s = s.replace(',', '')  # 1,234.5 style
            elif s.count(',') == 1 and '.' not in s:
                s = s.replace(',', '.')  # decimal comma: 1,5
            elif ',' in s:
                s = s.replace(',', '')
            value = Decimal(s)
        except InvalidOperation:
            return raw
        return format(value.normalize(), 'f') + ('%' if percent else '')

    @classmethod
    def _numbers(cls, text: str) -> List[str]:
        return [cls._canonical_number(m) for m in NUMBER_RE.findall(text.translate(_NUMBER_NORMALIZE))]

    @staticmethod
    def _technical_tokens(text: str) -> List[str]:
        # The pattern's trailing class can swallow a sentence-final
        # '.' or '-'; those are punctuation, not part of the token.
        return [tok.rstrip('.-') for tok in TECH_TOKEN_RE.findall(text) if tok.rstrip('.-')]

    @staticmethod
    def _script_residuals(text: str, target_key: str) -> List[str]:
        """Distinctive wrong-script characters left in the output."""
        if target_key in {'tw', 'hk', 'hant'}:
            bad = SIMPLIFIED_ONLY_CHARS
        elif target_key == 'cn':
            bad = TRADITIONAL_ONLY_CHARS
        else:
            return []
        return sorted({ch for ch in text if ch in bad})

    def _expected_glossary_form(self, target_term: str, target_key: str) -> str:
        """Glossary target as it should appear after script normalization."""
        config = TARGETS[target_key]['opencc']
        if not config:
            return target_term
        converter = self._opencc(config)
        if converter is None:
            return target_term
        try:
            return converter.convert(target_term)
        except Exception:
            return target_term

    @staticmethod
    def _status_from_checks(checks: List[Check]) -> str:
        if any(c.level == 'hold' for c in checks):
            return 'HOLD'
        if any(c.level == 'review' for c in checks):
            return 'REVIEW'
        return 'READY'

    def _detect_chinese_script(self, text: str) -> str:
        """Best-effort Hans/Hant detection using OpenCC idempotence."""
        t2s = self._opencc('t2s.json')
        s2t = self._opencc('s2t.json')
        if not t2s or not s2t:
            return 'zho_Hans'
        sample = re.sub(r'\s+', '', text[:5000])
        if not sample:
            return 'zho_Hans'
        simp = t2s.convert(sample)
        trad = s2t.convert(sample)
        simp_changes = sum(a != b for a, b in zip(sample, simp)) + abs(len(sample) - len(simp))
        trad_changes = sum(a != b for a, b in zip(sample, trad)) + abs(len(sample) - len(trad))
        # Traditional input needs more t2s changes; simplified input needs more s2t changes.
        return 'zho_Hant' if simp_changes > trad_changes else 'zho_Hans'

    def detect_source(self, text: str) -> Tuple[str, str, bool]:
        import langid
        iso, _score = langid.classify(text[:5000])
        if iso == 'zh':
            code = self._detect_chinese_script(text)
            return code, NLLB_TO_LABEL[code], True
        if iso not in LANGS:
            raise ValueError(f'自動判定した言語 {iso!r} は初版の対応表にありません。入力言語を手動指定してください。')
        code, label = LANGS[iso]
        return code, label, True

    def _translate_nllb(self, text: str, src_code: str, tgt_code: str) -> Tuple[str, int]:
        self._load_model()
        assert self.tokenizer is not None and self.model is not None
        self.tokenizer.src_lang = src_code
        encoded = self.tokenizer(text, return_tensors='pt', truncation=False)
        token_count = int(encoded['input_ids'].shape[1])
        if token_count > 500:
            raise ValueError(f'1文が {token_count} トークンあり、精度優先上限 500 を超えています。文を分割してください。')
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        target_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        import torch
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                do_sample=False,
                num_beams=5,
                repetition_penalty=1.05,
                max_new_tokens=min(512, max(64, token_count * 3)),
            )
        out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return out, token_count

    def _normalize_target_variant(self, text: str, target_key: str) -> Tuple[str, List[Check]]:
        cfg = TARGETS[target_key]
        checks: List[Check] = []
        config = cfg['opencc']
        if not config:
            checks.append(Check('ok', 'SCRIPT_NORMALIZATION', 'NLLB の繁體字出力をそのまま使用しました。'))
            return text, checks
        converter = self._opencc(config)
        if converter is None:
            checks.append(Check('review', 'OPENCC', f'OpenCC {config} を実行できず、地域表記の正規化を省略しました。'))
            return text, checks
        converted = converter.convert(text)
        if converted != text:
            checks.append(Check('ok', 'SCRIPT_NORMALIZATION', f'OpenCC {config} で文字・地域表記を正規化しました。'))
        else:
            checks.append(Check('ok', 'SCRIPT_NORMALIZATION', f'OpenCC {config} で追加変換は不要でした。'))
        return converted, checks

    def _chinese_checks(self, target: str, style: str, target_key: str) -> Tuple[List[Check], dict]:
        checks: List[Check] = []
        analysis = {'sentences': [], 'tokens': [], 'han_count': 0}

        han_count = len(HAN_RE.findall(target))
        nonspace_count = len(re.sub(r'\s+', '', target))
        analysis['han_count'] = han_count
        if nonspace_count and han_count / nonspace_count < 0.25:
            checks.append(Check('review', 'CHINESE_DENSITY', '出力中の漢字比率が低めです。固有名詞・記号中心の文でなければ要確認です。'))
        else:
            checks.append(Check('ok', 'CHINESE_DENSITY', '中国語文字を十分に含む出力です。'))

        if re.search(r'[\u3400-\u9fff][,;:][\u3400-\u9fff]', target) or re.search(r'[\u3400-\u9fff][.!?]', target):
            checks.append(Check('review', 'PUNCT_STYLE', '中国語本文中に半角の欧文句読点があります。中文句読点を確認してください。'))
        else:
            checks.append(Check('ok', 'PUNCT_STYLE', '明白な半角句読点混在は検出しませんでした。'))

        repeated = re.search(r'([\u3400-\u9fff])\1{2,}', target)
        if repeated:
            checks.append(Check('review', 'CHAR_REPEAT', f'同一漢字「{repeated.group(1)}」が3回以上連続しています。意図した表現か確認してください。'))

        informal = sorted({w for w in FORMAL_WARN_WORDS if w in target})
        if style in {'formal', 'wikipedia'} and informal:
            checks.append(Check('review', 'REGISTER', 'フォーマル文体では口語・ネット語寄りの表現を確認してください: ' + ', '.join(informal)))
        else:
            checks.append(Check('ok', 'REGISTER', '指定文体に対する明白な口語警告なし。'))

        stanza_lang = TARGETS[target_key]['stanza']
        if not self._load_stanza(stanza_lang):
            checks.append(Check('review', 'STANZA', f'Stanza の {stanza_lang} 中国語解析を実行できませんでした。'))
            return checks, analysis

        try:
            pipe = self.stanza_pipelines[stanza_lang]
            doc = pipe(target)
            for sent in doc.sentences:
                roots = [w for w in sent.words if w.head == 0]
                analysis['sentences'].append({
                    'text': sent.text,
                    'root': roots[0].text if roots else '',
                    'root_upos': roots[0].upos if roots else '',
                })
                for w in sent.words:
                    analysis['tokens'].append({
                        'text': w.text,
                        'lemma': w.lemma,
                        'upos': w.upos,
                        'head': w.head,
                        'deprel': w.deprel,
                    })
            if doc.sentences and all(any(w.head == 0 for w in s.words) for s in doc.sentences):
                checks.append(Check('ok', 'STANZA', '全出力文で依存構造の root を取得できました。'))
            else:
                checks.append(Check('review', 'STANZA', '依存構造解析が不完全な文があります。'))
        except Exception as exc:
            checks.append(Check('review', 'STANZA', f'中国語構文解析に失敗しました: {exc}'))
        return checks, analysis

    def translate_sentence(
        self,
        sentence: str,
        src_code: str,
        target_key: str,
        style: str,
        glossary: Dict[str, str],
        run_backtranslation: bool,
    ) -> dict:
        checks: List[Check] = []
        target_cfg = TARGETS[target_key]
        target_code = target_cfg['code']

        protected, protected_map = self._protect(sentence, 'PROT', PROTECT_PATTERNS)
        prepared, glossary_map = self._protect_glossary(protected, glossary)
        source_prose = TOKEN_RE.sub('', prepared)
        source_numbers = self._numbers(source_prose)
        source_tech = self._technical_tokens(source_prose)
        prepared, tech_map = self._protect(prepared, 'TECH', [TECH_TOKEN_RE.pattern])
        all_map = {**protected_map, **glossary_map, **tech_map}

        try:
            # Chinese-to-Chinese is script/regional conversion, not paraphrasing MT.
            if src_code in {'zho_Hans', 'zho_Hant'}:
                target_raw = prepared
                token_count = len(sentence)
                checks.append(Check('ok', 'ZH_TO_ZH', '中国語入力なので NLLB 再翻訳を避け、表記変換を優先しました。'))
            else:
                target_raw, token_count = self._translate_preserving(prepared, all_map, src_code, target_code)
                if all_map and token_count:
                    checks.append(Check('review', 'SEGMENTED_TRANSLATION', 'URL・記法・固定用語・技術名を保持して本文を分割翻訳しました。前後の語順・文脈を確認してください。'))
        except ValueError as exc:
            checks.append(Check('hold', 'MODEL_INPUT', str(exc)))
            return {
                'source': sentence,
                'translation': '',
                'backtranslation': '',
                'roundtrip_similarity': None,
                'source_tokens': None,
                'status': 'HOLD',
                'checks': [asdict(c) for c in checks],
                'analysis': {},
            }

        if not target_raw.strip():
            checks.append(Check('hold', 'EMPTY_TRANSLATION', '翻訳モデルが空の訳文を返しました。'))

        # Normalize prose BEFORE restoring literal URLs/wiki markup/technical
        # names. Glossary targets alone are normalized to the requested script.
        normalized, variant_checks = self._normalize_target_variant(target_raw, target_key)
        checks.extend(variant_checks)
        normalized_glossary = {key: self._expected_glossary_form(value, target_key)
                               for key, value in glossary_map.items()}
        restore_map = {**protected_map, **normalized_glossary, **tech_map}
        target, missing = self._restore(normalized, restore_map)
        if missing:
            checks.append(Check('hold', 'PLACEHOLDER_LOSS', '保護した Wiki 記法・URL・用語集トークンの一部が翻訳中に失われました。'))
        else:
            checks.append(Check('ok', 'PLACEHOLDER_LOSS', '保護トークンをすべて復元しました。'))

        # Wrong-script residue (e.g. Simplified characters in a Taiwan
        # Traditional output) is a failed normalization, not a nuance.
        checked_prose = normalized
        for key in protected_map:
            checked_prose = checked_prose.replace(key, '')
        for key, value in normalized_glossary.items():
            checked_prose = checked_prose.replace(key, value)
        for key in tech_map:
            checked_prose = checked_prose.replace(key, '')
        residuals = self._script_residuals(checked_prose, target_key)
        if residuals:
            script_name = '簡体字' if target_key == 'cn' else '繁體字'
            wrong_name = '繁體字' if target_key == 'cn' else '簡体字'
            checks.append(Check('hold', 'SCRIPT_MISMATCH', f'{script_name}指定なのに{wrong_name}固有の文字が残っています: ' + ', '.join(residuals)))

        # <ref> tags are ASCII, so their count survives script
        # normalization. This catches a lost ref even if it was never
        # placeholder-protected (e.g. an unmatched / malformed tag).
        source_ref_count = len(re.findall(r'<ref\b', sentence))
        target_ref_count = len(re.findall(r'<ref\b', target))
        if target_ref_count < source_ref_count:
            checks.append(Check('hold', 'REF_LOSS', f'<ref> タグが原文 {source_ref_count} 件から {target_ref_count} 件に減っています。'))

        expected_literals = {literal: sentence.count(literal) for literal in protected_map.values()}
        if any(target.count(literal) != count for literal, count in expected_literals.items()):
            checks.append(Check('hold', 'PROTECTED_CONTENT', '保護したURL・Wiki記法の内容または出現回数が一致しません。'))

        # Glossary terms are placeholder-protected (loss there is
        # already HOLD), but verify the final, script-normalized
        # output as well: a restored term must still be present in
        # the form normalization would produce for it.
        missing_glossary = []
        for expected, count in Counter(normalized_glossary.values()).items():
            if target.count(expected) < count:
                missing_glossary.append(expected)
        if missing_glossary:
            checks.append(Check('hold', 'GLOSSARY_LOSS', '用語集で指定した訳語が出力にありません: ' + ', '.join(missing_glossary)))

        target_numbers = self._numbers(TOKEN_RE.sub('', normalized))
        if sorted(source_numbers) != sorted(target_numbers):
            checks.append(Check('hold', 'NUMBER_PRESERVATION', f'数値が一致しません: source={source_numbers} target={target_numbers}'))
        else:
            checks.append(Check('ok', 'NUMBER_PRESERVATION', '数値を保持しています。'))

        if any(key in target and key not in sentence for key in all_map):
            checks.append(Check('hold', 'UNRESTORED_TOKEN', '未復元の内部トークンが出力に残っています。'))

        output_tech = Counter(self._technical_tokens(TOKEN_RE.sub('', normalized)))
        output_tech.update(tok for key, value in tech_map.items() if key in normalized
                           for tok in self._technical_tokens(value))
        missing_tech = [tok for tok, count in Counter(source_tech).items() if output_tech[tok] != count]
        if missing_tech:
            # A dropped proper name / model name (OpenAI, GPT4, HTML5)
            # changes the facts of the sentence, like a dropped
            # number. It was previously only REVIEW -- and, due to
            # the \b bug in TECH_TOKEN_RE, usually not even detected
            # in Japanese text at all.
            checks.append(Check('hold', 'TECH_TOKEN', '英数字の技術名・略語・固有名詞が出力から消えています: ' + ', '.join(missing_tech)))
        elif source_tech:
            checks.append(Check('ok', 'TECH_TOKEN', '検出した英数字の技術名・略語を保持しています。'))

        if checked_prose.strip() or normalized_glossary:
            zh_checks, analysis = self._chinese_checks(target, style, target_key)
            checks.extend(zh_checks)
        else:
            analysis = {'sentences': [], 'tokens': [], 'han_count': 0}

        back = ''
        sim = None
        if run_backtranslation and src_code not in {'zho_Hans', 'zho_Hant'}:
            try:
                back_protected, bt_map = self._protect(target, 'PROT', PROTECT_PATTERNS)
                back_raw, _ = self._translate_preserving(back_protected, bt_map, target_code, src_code)
                back, bt_missing = self._restore(back_raw, bt_map)
                if bt_missing:
                    checks.append(Check('review', 'BACK_PLACEHOLDER', '逆翻訳時に保護記法の復元失敗がありました。'))
                sim = self.roundtrip_similarity(sentence, back)
                if sim < 30:
                    checks.append(Check('hold', 'ROUNDTRIP', f'逆翻訳一致度が低いです ({sim}%). 意味の脱落・変形を要確認。'))
                elif sim < 55:
                    checks.append(Check('review', 'ROUNDTRIP', f'逆翻訳一致度は中程度です ({sim}%).'))
                else:
                    checks.append(Check('ok', 'ROUNDTRIP', f'逆翻訳一致度 {sim}%.'))
            except Exception as exc:
                checks.append(Check('review', 'ROUNDTRIP', f'逆翻訳を実行できませんでした: {exc}'))
        elif src_code in {'zho_Hans', 'zho_Hant'}:
            checks.append(Check('ok', 'ROUNDTRIP', '中国語から中国語への表記変換のため逆翻訳を省略しました。'))

        return {
            'source': sentence,
            'translation': target,
            'backtranslation': back,
            'roundtrip_similarity': sim,
            'source_tokens': token_count,
            'status': self._status_from_checks(checks),
            'checks': [asdict(c) for c in checks],
            'analysis': analysis,
        }

    def translate_document(
        self,
        text: str,
        source: str = 'auto',
        target: str = 'cn',
        style: str = 'formal',
        glossary_text: str = '',
        run_backtranslation: bool = True,
    ) -> dict:
        if not text.strip():
            raise ValueError('入力がありません。')
        if target not in TARGETS:
            raise ValueError('未対応の中国語出力モードです。')
        if style not in {'formal', 'wikipedia', 'neutral'}:
            raise ValueError('未対応の文体です。')

        auto_detected = False
        if source == 'auto':
            src_code, src_label, auto_detected = self.detect_source(text)
        else:
            valid_codes = {v[0] for v in LANGS.values()} | {'zho_Hans', 'zho_Hant'}
            if source not in valid_codes:
                raise ValueError('未対応の入力言語コードです。')
            src_code = source
            src_label = NLLB_TO_LABEL.get(src_code, src_code)

        glossary = self.parse_glossary(glossary_text)
        sentences = self.split_sentences(text)
        results = [
            self.translate_sentence(s, src_code, target, style, glossary, run_backtranslation)
            for s in sentences
        ]

        if any(r['status'] == 'HOLD' for r in results):
            overall = 'HOLD'
        elif any(r['status'] == 'REVIEW' for r in results) or auto_detected:
            overall = 'REVIEW'
        else:
            overall = 'READY'

        document_checks = []
        if auto_detected:
            document_checks.append({
                'level': 'review',
                'code': 'AUTO_LANGUAGE',
                'message': f'入力言語を {src_label} と自動判定しました。公開用なら手動指定を推奨します。',
            })

        target_cfg = TARGETS[target]
        return {
            'status': overall,
            'source_language': {'code': src_code, 'label': src_label, 'auto': auto_detected},
            'target_language': {'key': target, 'code': target_cfg['code'], 'label': target_cfg['label']},
            'style': style,
            'engine': MODEL_NAME,
            'sentence_count': len(results),
            'translation': '\n'.join(r['translation'] for r in results if r['translation']),
            'document_checks': document_checks,
            'sentences': results,
        }
