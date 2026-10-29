from __future__ import annotations

import os
import threading
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from translator import AccuracyFirstChineseTranslator, EngineRequestError, InsufficientMemoryError, ModelUnavailableError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024
translator = AccuracyFirstChineseTranslator()
translation_lock = threading.Lock()


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return jsonify({'error': '入力データが大きすぎます。文章を分けて送信してください。'}), 413


@app.get('/')
def index():
    return render_template(
        'index.html',
        languages=translator.language_choices(),
        targets=translator.target_choices(),
    )


@app.get('/api/health')
def health():
    return jsonify({
        'ok': True,
        'model_loaded': translator.model_loaded,
        'model_state': translator.model_state,
        'engine': translator._engine(),
        'memory_limit_mb': translator.memory_limit_mb,
        'translation_ready': translator.engine_ready(),
        'busy': translation_lock.locked(),
        'revision': os.getenv('RENDER_GIT_COMMIT', ''),
        'stanza_loaded': sorted(translator.stanza_pipelines.keys()),
        'targets': {k: v['code'] for k, v in __import__('translator').TARGETS.items()},
    })


@app.post('/api/translate')
def translate_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'JSONオブジェクトを送信してください。'}), 400
    for key in ('text', 'source', 'target', 'style', 'glossary'):
        if key in data and not isinstance(data[key], str):
            return jsonify({'error': f'{key} は文字列で指定してください。'}), 400
    if 'backtranslation' in data and not isinstance(data['backtranslation'], bool):
        return jsonify({'error': 'backtranslation は true または false で指定してください。'}), 400
    text = data.get('text', '').strip()
    source = data.get('source', 'auto')
    target = data.get('target', 'cn')
    style = data.get('style', 'formal')
    glossary = data.get('glossary', '')
    run_backtranslation = data.get('backtranslation', True)

    if not text:
        return jsonify({'error': '入力がありません。'}), 400
    if len(text) > 20000 or len(glossary) > 20000:
        return jsonify({'error': '原文・用語集はそれぞれ20,000文字以内にしてください。'}), 400

    # The tokenizer's source language and model/parser instances are shared.
    # Reject overlapping work instead of corrupting languages or duplicating RAM.
    if not translation_lock.acquire(blocking=False):
        return jsonify({'error': '別の翻訳を処理中です。完了してから再実行してください。', 'code': 'BUSY'}), 429, {'Retry-After': '5'}

    try:
        result = translator.translate_document(
            text=text,
            source=source,
            target=target,
            style=style,
            glossary_text=glossary,
            run_backtranslation=run_backtranslation,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except InsufficientMemoryError as exc:
        return jsonify({'error': str(exc), 'code': 'INSUFFICIENT_MEMORY'}), 503
    except EngineRequestError as exc:
        return jsonify({'error': str(exc), 'code': 'ENGINE_REQUEST_FAILED'}), 503
    except ModelUnavailableError:
        app.logger.exception('Translation model could not be loaded')
        return jsonify({
            'error': '翻訳モデルを読み込めません。サーバーで python scripts/setup_models.py を実行し、モデルの保存先・通信・空きメモリを確認してください。',
            'code': 'MODEL_UNAVAILABLE',
        }), 503
    except Exception:
        app.logger.exception('Translation failed')
        return jsonify({'error': '翻訳処理でエラーが発生しました。サーバーログを確認してください。'}), 500
    finally:
        translation_lock.release()


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0' if 'PORT' in os.environ else '127.0.0.1')
    app.run(host=host, port=port, debug=os.getenv('FLASK_DEBUG') == '1')
