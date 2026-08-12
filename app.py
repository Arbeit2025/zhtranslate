from __future__ import annotations

import os
from flask import Flask, jsonify, render_template, request

from translator import AccuracyFirstChineseTranslator

app = Flask(__name__)
translator = AccuracyFirstChineseTranslator()


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
        'stanza_loaded': sorted(translator.stanza_pipelines.keys()),
        'targets': {k: v['code'] for k, v in __import__('translator').TARGETS.items()},
    })


@app.post('/api/translate')
def translate_api():
    data = request.get_json(silent=True) or {}
    text = str(data.get('text', '')).strip()
    source = str(data.get('source', 'auto'))
    target = str(data.get('target', 'cn'))
    style = str(data.get('style', 'formal'))
    glossary = str(data.get('glossary', ''))
    run_backtranslation = bool(data.get('backtranslation', True))

    if not text:
        return jsonify({'error': '入力がありません。'}), 400

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
    except Exception as exc:
        return jsonify({'error': f'翻訳処理でエラーが発生しました: {exc}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='127.0.0.1', port=port, debug=os.getenv('FLASK_DEBUG') == '1')
