"""Run repeatable checks against real NLLB, Stanza, OpenCC and Flask.

Install requirements-dev.txt and run setup_models.py first. This is separate
from the lightweight CI; it requires model files and several GB of RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['AUTO_DOWNLOAD_STANZA'] = '0'
os.environ.pop('DISABLE_STANZA', None)

from app import app, translator


def contains(*parts):
    return lambda data: all(part in data['translation'] for part in parts)


def cases():
    return [
        ('Japanese and backtranslation', {'text': '今日は天気がよいです。', 'source': 'jpn_Jpan', 'backtranslation': True},
         lambda d: contains('今天', '天气')(d) and bool(d['sentences'][0]['backtranslation'])),
        ('English and backtranslation', {'text': 'The capital of China is Beijing.', 'source': 'eng_Latn', 'backtranslation': True},
         lambda d: contains('中国', '北京')(d) and bool(d['sentences'][0]['backtranslation'])),
        ('Numbers', {'text': '2026年に12冊の本を読んだ。', 'source': 'jpn_Jpan'}, contains('2026', '12')),
        ('Fixed glossary', {'text': '自己調整学習は学習を支援する。', 'source': 'jpn_Jpan', 'glossary': '自己調整学習=自我调节学习'}, contains('自我调节学习')),
        ('CJK adjacent names', {'text': 'OpenAIはGPT4とHTML5を使用する。', 'source': 'jpn_Jpan'}, contains('OpenAI', 'GPT4', 'HTML5')),
        ('URL query', {'text': '詳細は https://example.com/search?q=test&x=1 を参照。', 'source': 'jpn_Jpan'}, contains('https://example.com/search?q=test&x=1')),
        ('Reference punctuation', {'text': '中国は東アジアに位置する国である。<ref>出典。補足。</ref>', 'source': 'jpn_Jpan'}, contains('<ref>出典。補足。</ref>')),
        ('Wiki link', {'text': '[[中国]]の首都は北京である。', 'source': 'jpn_Jpan'}, contains('[[中国]]', '北京')),
        *[(f'English to {target}', {'text': 'This software runs on the network.', 'source': 'eng_Latn', 'target': target},
           lambda d: contains('這個', '軟體')(d) and '软件' not in d['translation']) for target in ('tw', 'hk', 'hant')],
        ('Simplified to Taiwan', {'text': '这个软件在网络上运行。', 'source': 'zho_Hans', 'target': 'tw'}, contains('這個', '軟體', '網路')),
        ('Overlong sentence', {'text': ('word ' * 520) + '.', 'source': 'eng_Latn'},
         lambda d: d['status'] == 'HOLD' and not d['translation']),
        ('Unicode URL and nested wiki', {'text': '参照 https://example.com/體育?q=1&x=2 {{outer|x={{inner|體育}}}} 。', 'source': 'jpn_Jpan'},
         contains('https://example.com/體育?q=1&x=2', '{{outer|x={{inner|體育}}}}')),
        ('Repeated glossary', {'text': '自己調整学習と自己調整学習を説明する。', 'source': 'jpn_Jpan', 'glossary': '自己調整学習=自我调节学习'},
         lambda d: d['translation'].count('自我调节学习') == 2),
        ('Chinese glossary', {'text': '学习这个软件。', 'source': 'zho_Hans', 'target': 'tw', 'glossary': '学习=研修'}, contains('研修', '這個', '軟體')),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('model-verification.json'))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    rows = []
    for round_ in range(1, args.repeat + 1):
        with app.test_client() as client:
            for name, payload, verify in cases():
                start = time.monotonic()
                payload = {'target': 'cn', 'style': 'formal', 'backtranslation': False, **payload}
                response = client.post('/api/translate', json=payload)
                data = response.get_json()
                expected_hold = name == 'Overlong sentence'
                passed = (response.status_code == 200 and verify(data)
                          and (expected_hold or data['status'] != 'HOLD'))
                row = {'round': round_, 'name': name, 'passed': bool(passed),
                       'http_status': response.status_code, 'seconds': round(time.monotonic() - start, 3),
                       'input': payload, 'result': data}
                rows.append(row)
                args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({k: v for k, v in row.items() if k not in {'input', 'result'}}, ensure_ascii=False), flush=True)
    models_ready = translator.model_loaded and {'zh-hans', 'zh-hant'} <= translator.stanza_pipelines.keys()
    print(json.dumps({'total': len(rows), 'passed': sum(row['passed'] for row in rows),
                      'model_loaded': translator.model_loaded, 'stanza_loaded': sorted(translator.stanza_pipelines)}), flush=True)
    return 0 if models_ready and all(row['passed'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
