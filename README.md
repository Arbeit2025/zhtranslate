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
  -> Wiki markup / URL protection before sentence segmentation
  -> glossary / technical-name protection (one key per occurrence)
  -> NLLB translation of prose spans -> zho_Hans / zho_Hant
  -> OpenCC script / regional normalization
  -> literal restoration, then preservation / number / technical-token checks
  -> Chinese Stanza POS / lemma / dependency parse
  -> register / punctuation / output-shape checks
  -> optional Chinese -> source back-translation
  -> per-sentence READY / REVIEW / HOLD
```

中国語から中国語への入力では、不必要な言い換えを避けるため原則として NLLB に再翻訳させず、OpenCC の表記変換を優先します。

## Status policy

- `READY`: このプログラム内の必須検査を通過。
- `REVIEW`: 自動言語判定、構文解析未実行、地域表記正規化未実行、中程度の逆翻訳一致度、口語・ネット語など確認点あり。
- `HOLD`: 数値不一致、保護記法・URL・`<ref>` の欠落、用語集で指定した訳語の欠落、技術名・固有名詞（`OpenAI` 等）の消失、指定と異なる簡繁字の残留、500 NLLB tokens 超の単文、極端に低い逆翻訳一致度など。

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

### 起動しない・翻訳が終わらない場合

画面表示と翻訳モデルの準備は別です。`/api/health` の `ok: true` はHTTPプロセスの稼働を表し、`translation_ready` / `model_loaded` が実モデルの読込完了を表します。モデルは最初の翻訳時に読み込みます。`model_state` は `not_loaded / loading / ready / error / insufficient_memory`、`busy` は処理中、`memory_limit_mb` は検出できたメモリ上限、`revision` はRenderが設定した公開コミットです。中国語同士の表記変換にはNLLBを使いません。

- 依存関係導入後、**アプリと同じユーザー・モデル保存先**で `python scripts/setup_models.py` を実行してください。`MT_MODEL` を設定している場合はセットアップにも同じ値を使用します。
- NLLBは600Mパラメータのモデルです。FP32の重みだけで約2.4GBあり、実行時はさらにPython・生成処理・Stanzaのメモリが必要です。512MBの環境には収まりません。複数のワーカーで同じモデルをロードしない構成にしてください。
- モデルが読み込めなければAPIは `503 MODEL_UNAVAILABLE` を返します。通信・空きメモリ・キャッシュ配置を確認してください。OSによるメモリ不足の強制終了はPythonでは捕捉できず、公開先のログ確認が必要です。
- 既定のNLLBモデルで検出したメモリ上限が3GiB未満の場合は、重いライブラリの読み込み前に `503 INSUFFICIENT_MEMORY` を返してプロセスの強制終了を避けます。これは最低限のガードで、3GiBなら十分という動作保証ではありません。2GiB未満の環境ではStanzaも読み込まず、中国語同士の表記変換を `REVIEW` で提供します。
- Stanzaは事前ダウンロードを基本とし、リクエスト中の自動ダウンロードは既定で無効です。有効にする場合だけ `AUTO_DOWNLOAD_STANZA=1` を指定してください。解析器が使えなければ `REVIEW` として表示します。
- 同時翻訳は1件ずつ処理します。処理中に別の翻訳を送ると `429 BUSY` です。原文と用語集は各20,000文字まで受け付けます。

Renderなどで `python app.py` を起動する場合、`PORT` が設定されていれば `0.0.0.0` で待ち受けます。ローカルでは既定で `127.0.0.1`、必要なら `HOST` で指定できます。既存の `gunicorn app:app` 起動では `gunicorn.conf.py` が自動適用され、`0.0.0.0:$PORT`、ワーカー1個・4スレッド・タイムアウト300秒になります。

## Accuracy features

### 1. Direct multilingual -> Chinese

日本語は `jpn_Jpan`、英語は `eng_Latn`、出力は `zho_Hans` / `zho_Hant` を使います。

### 2. Simplified / Traditional / regional forms

OpenCC により次を使用します。

- 中国大陸簡体字: `t2s.json`
- 台湾正体字: `s2twp.json`
- 香港繁体字: `s2hk.json`
- 繁體字（通用）: `s2t.json`

`t2tw.json` / `t2hk.json` は入力が既に繁體字である前提で、簡体字が混入しても変換しません。台湾・香港向けは簡体字からの変換を含む `s2twp.json` / `s2hk.json` を使い、繁體字入力に対しても冪等です。正規化後も指定と異なる字形固有の文字が残っていれば `HOLD`（`SCRIPT_MISMATCH`）にします。

OpenCC は翻訳モデルではなく、簡繁字・異体字・地域用語の変換器として使用します。

### 3. Long text

文章全体を一発でモデルへ投げず、文単位で処理します。単文が 500 NLLB tokens を超えた場合は `HOLD` にします。文分割は URL・`<ref>...</ref>`・テンプレート・Wikiリンクを保護してから行うため、URL の `?` 以降のクエリや、句点を含む `<ref>` の内側で文が切れることはありません。

### 4. Preservation checks

- 数値
- URL
- `<ref>...</ref>`
- `{{template}}`
- `[[wikilink]]`
- `[12]` 型の参照
- ユーザー固定用語集（最終出力に指定訳語がなければ `HOLD`）
- `OpenAI`, `HTML5`, `GPT4` のような英数字技術名・固有名詞の消失（日本語に隣接していても検出し、消失は `HOLD`）

保護対象や用語集の内部トークンをNLLBにそのままコピーさせる方式は使いません。本文の区間だけを翻訳し、表記変換後にURL・Wiki記法・技術名をそのまま復元します。ネストしたテンプレートと内部トークンに似た原文も保持します。用語集は出現ごとに扱い、中国語訳語を指定した簡繁字へ正規化します。分割翻訳では前後の語順・文脈が不自然になる場合があるため、`SEGMENTED_TRANSLATION` の `REVIEW` を必ず付けます。Wiki構文全般の完全対応や意味の完全保存を保証するものではありません。

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

モデルをロードしない軽量検査（pytest は開発用依存関係として requirements-dev.txt に分離しています）:

```bash
pip install -r requirements-test.txt
pytest -q
```

この軽量テストは実OpenCC・Flaskと、翻訳出力の故障注入を使い、モデル本体の翻訳精度は検証しません。GitHub Actionsでも同じテストを実行します。実モデルを含む確認は `requirements-dev.txt` とモデルを準備した上で、`python scripts/verify_models.py --repeat 3 --output model-verification.json` を実行してください。

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
