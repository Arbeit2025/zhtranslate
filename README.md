# Chinese Accuracy-First MT v1

多言語の文章を **現代中国語**へ翻訳し、文ごとに `READY / REVIEW / HOLD` を返すローカル翻訳ワークベンチです。

前作の **Volapük Accuracy-First MT** の「未知・曖昧な箇所を流暢さで隠さず、検査結果を明示する」という思想を引き継ぎ、中国語ではルールだけで全文生成せず、NMT + 中国語側の検査器を組み合わせています。

## Output modes

- `cn`: 简体中文（中国大陆）
- `tw`: 繁體中文（台灣正體）
- `hk`: 繁體中文（香港）
- `hant`: 繁體中文（通用）

NLLB の `zho_Hans` / `zho_Hant` で基礎翻訳し、必要に応じて OpenCC で字形・地域表記を正規化します。

## Architecture

```text
source text
  -> source-language selection / detection
  -> sentence segmentation
  -> Wiki markup / URL / glossary protection
  -> direct NLLB translation -> zho_Hans / zho_Hant
  -> OpenCC script / regional normalization
  -> protected-token + number + technical-token checks
  -> Chinese Stanza POS / lemma / dependency parse
  -> register / punctuation / output-shape checks
  -> optional Chinese -> source back-translation
  -> per-sentence READY / REVIEW / HOLD
```

中国語から中国語への入力では、不必要な言い換えを避けるため原則として NLLB に再翻訳させず、OpenCC の表記変換を優先します。

## Status policy

- `READY`: このプログラム内の必須検査を通過。
- `REVIEW`: 自動言語判定、構文解析未実行、地域表記正規化未実行、中程度の逆翻訳一致度、口語・ネット語、技術名の消失など確認点あり。
- `HOLD`: 数値不一致、保護記法の欠落、500 NLLB tokens 超の単文、極端に低い逆翻訳一致度など。

`READY` は正訳証明・公的認証ではありません。

## Install

Python 3.11+ 推奨。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/setup_models.py
python app.py
```

ブラウザで `http://127.0.0.1:5000` を開きます。

初回は NLLB と Stanza の中国語モデルを取得するため、大きな通信が発生します。

## Accuracy features

### 1. Direct multilingual -> Chinese

日本語は `jpn_Jpan`、英語は `eng_Latn`、出力は `zho_Hans` / `zho_Hant` を使います。

### 2. Simplified / Traditional / regional forms

OpenCC により次を使用します。

- 中国大陸簡体字: `t2s.json`
- 台湾正体字: `t2tw.json`
- 香港繁体字: `t2hk.json`

OpenCC は翻訳モデルではなく、簡繁字・異体字・地域用語の変換器として使用します。

### 3. Long text

文章全体を一発でモデルへ投げず、文単位で処理します。単文が 500 NLLB tokens を超えた場合は `HOLD` にします。

### 4. Preservation checks

- 数値
- URL
- `<ref>...</ref>`
- `{{template}}`
- `[[wikilink]]`
- `[12]` 型の参照
- ユーザー固定用語集
- `OpenAI`, `HTML5`, `GPT4` のような英数字技術名の消失警告

### 5. Chinese-side parser

Stanza の `zh-hans` / `zh-hant` と `tokenize,pos,lemma,depparse` を使います。依存構造 root が得られるかを確認しますが、構文解析成功だけで文法的正しさを認定しません。

### 6. Back-translation

任意で中国語から原言語へ逆翻訳し、元文との文字列類似度を補助警報として使います。同一モデルを往復に使うため、相関した誤りを見逃すことがあります。

### 7. Fixed glossary

```text
自己調整学習=自我调节学习
人工知能=人工智能
OpenAI=OpenAI
```

台湾モードなら用語集自体を台湾表記で指定するのが最も確実です。

## Model / dependency note

NLLB-200 distilled 600M のモデルカードは `zho_Hans` と `zho_Hant` を収録し、モデルを CC-BY-NC-4.0 としています。また研究用途を主目的とし、本番配備・認証翻訳・長文書翻訳を想定しない旨を記載しています。そのため、このリポジトリにはモデル本体を同梱しません。

OpenCC は Apache-2.0 で、Python API と中国大陸・台湾・香港向けの標準設定を提供しています。

## Tests

モデルをロードしない軽量検査:

```bash
pytest -q
```

## Next candidates

- Wikidata sitelink を使った日本語固有名詞 -> 中国語記事名の解決
- 中国語 Wikipedia の表記・句読法チェック
- 独立した第2翻訳エンジンとのクロスチェック
- 中国語 Wikipedia / 公的用語集を使った用語候補検証
- 人名・地名の「漢字表記 / 音訳 / 原綴り保持」の選択器
- 日付・数量詞・単位の意味保存チェック強化
- 翻訳メモリ（TM）

## Primary references

- NLLB model card: https://huggingface.co/facebook/nllb-200-distilled-600M
- Stanza: https://stanfordnlp.github.io/stanza/
- OpenCC: https://github.com/BYVoid/OpenCC
