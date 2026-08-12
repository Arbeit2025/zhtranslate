from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple


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
        'opencc': 't2tw.json',
    },
    'hk': {
        'code': 'zho_Hant',
        'label': '繁體中文（香港）',
        'stanza': 'zh-hant',
        'opencc': 't2hk.json',
    },
    'hant': {
        'code': 'zho_Hant',
        'label': '繁體中文（通用）',
        'stanza': 'zh-hant',
        'opencc': None,
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
    r'https?://[^\s<>]+',
    r'<ref\b[^>]*>[\s\S]*?</ref>',
    r'<ref\b[^>]*/>',
    r'\{\{[^{}]{0,2000}\}\}',
    r'\[\[[^\[\]]{0,2000}\]\]',
    r'\[[0-9]+\]',
]

NUMBER_RE = re.compile(r'(?<![0-9A-Za-z])[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?%?(?![0-9A-Za-z])')
TOKEN_RE = re.compile(r'ZZ(?:PROT|TERM)\d{3,5}ZZ')
HAN_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
TECH_TOKEN_RE = re.compile(r'\b(?:[A-Z]{2,}[A-Z0-9.-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+\d+[A-Za-z0-9.-]*|\d+[A-Za-z][A-Za-z0-9.-]*)\b')


@dataclass
class Check:
    level: str  # ok, review, hold
    code: str
    message: str


class AccuracyFirstChineseTranslator:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.stanza_pipelines: Dict[str, object] = {}
        self.opencc_converters: Dict[str, object] = {}

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
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            self.model = self.model.to('mps')
        self.model.eval()
        self.model_loaded = True

    def _load_stanza(self, lang: str) -> bool:
        if lang in self.stanza_pipelines:
            return True
        if os.getenv('DISABLE_STANZA') == '1':
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
                )
            except Exception:
                if os.getenv('AUTO_DOWNLOAD_STANZA', '1') != '1':
                    return False
                stanza.download(lang, verbose=False)
                pipe = stanza.Pipeline(
                    lang,
                    processors='tokenize,pos,lemma,depparse',
                    use_gpu=torch.cuda.is_available(),
                    verbose=False,
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
        out: List[str] = []
        for paragraph in re.split(r'\n{2,}', text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # Split Japanese/Chinese punctuation even with no following space.
            # Latin punctuation splits when followed by whitespace.
            parts = re.split(r'(?<=[。！？!?])|(?<=[.!?])\s+', paragraph)
            for p in parts:
                p = p.strip()
                if p:
                    out.append(p)
        return out or [text]

    @staticmethod
    def _protect(text: str, prefix: str, patterns: Iterable[str]) -> Tuple[str, Dict[str, str]]:
        mapping: Dict[str, str] = {}
        combined = re.compile('|'.join(f'(?:{p})' for p in patterns), re.I)

        def repl(match: re.Match) -> str:
            key = f'ZZ{prefix}{len(mapping):04d}ZZ'
            mapping[key] = match.group(0)
            return key

        return combined.sub(repl, text), mapping

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
        for src in sorted(glossary, key=len, reverse=True):
            if src not in text:
                continue
            key = f'ZZTERM{len(mapping):04d}ZZ'
            text = text.replace(src, key)
            mapping[key] = glossary[src]
        return text, mapping

    @staticmethod
    def _restore(text: str, mapping: Dict[str, str]) -> Tuple[str, List[str]]:
        missing = []
        for key, original in mapping.items():
            if key not in text:
                missing.append(key)
                continue
            text = text.replace(key, original)
        return text, missing

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
    def _numbers(text: str) -> List[str]:
        return NUMBER_RE.findall(text)

    @staticmethod
    def _technical_tokens(text: str) -> List[str]:
        return TECH_TOKEN_RE.findall(text)

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

        if re.search(r'[\u3400-\u9fff][,;:][\u3400-\u9fff]', target) or re.search(r'[\u3400-\u9fff][!?]', target):
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
        all_map = {**protected_map, **glossary_map}

        source_numbers = self._numbers(sentence)
        source_tech = self._technical_tokens(sentence)

        try:
            # Chinese-to-Chinese is script/regional conversion, not paraphrasing MT.
            if src_code in {'zho_Hans', 'zho_Hant'} and not glossary_map:
                target_raw = prepared
                token_count = len(sentence)
                checks.append(Check('ok', 'ZH_TO_ZH', '中国語入力なので NLLB 再翻訳を避け、表記変換を優先しました。'))
            else:
                target_raw, token_count = self._translate_nllb(prepared, src_code, target_code)
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

        target_restored, missing = self._restore(target_raw, all_map)
        if missing:
            checks.append(Check('hold', 'PLACEHOLDER_LOSS', '保護した Wiki 記法・URL・用語集トークンの一部が翻訳中に失われました。'))
        else:
            checks.append(Check('ok', 'PLACEHOLDER_LOSS', '保護トークンをすべて復元しました。'))

        target, variant_checks = self._normalize_target_variant(target_restored, target_key)
        checks.extend(variant_checks)

        target_numbers = self._numbers(target)
        if sorted(source_numbers) != sorted(target_numbers):
            checks.append(Check('hold', 'NUMBER_PRESERVATION', f'数値が一致しません: source={source_numbers} target={target_numbers}'))
        else:
            checks.append(Check('ok', 'NUMBER_PRESERVATION', '数値を保持しています。'))

        if TOKEN_RE.search(target):
            checks.append(Check('hold', 'UNRESTORED_TOKEN', '未復元の内部トークンが出力に残っています。'))

        missing_tech = [tok for tok in source_tech if tok not in target]
        if missing_tech:
            checks.append(Check('review', 'TECH_TOKEN', '英数字の技術名・略語が出力から消えています。訳語化が妥当か確認: ' + ', '.join(missing_tech)))
        elif source_tech:
            checks.append(Check('ok', 'TECH_TOKEN', '検出した英数字の技術名・略語を保持しています。'))

        zh_checks, analysis = self._chinese_checks(target, style, target_key)
        checks.extend(zh_checks)

        back = ''
        sim = None
        if run_backtranslation and src_code not in {'zho_Hans', 'zho_Hant'}:
            try:
                back_protected, bt_map = self._protect(target, 'PROT', PROTECT_PATTERNS)
                back_raw, _ = self._translate_nllb(back_protected, target_code, src_code)
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
        if target not in TARGETS:
            raise ValueError('未対応の中国語出力モードです。')

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
