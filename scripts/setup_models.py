import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import stanza

MODEL = os.getenv('MT_MODEL', 'facebook/nllb-200-distilled-600M')

print('Downloading NLLB tokenizer/model...')
AutoTokenizer.from_pretrained(MODEL)
AutoModelForSeq2SeqLM.from_pretrained(MODEL)

print('Downloading Stanza Chinese models...')
stanza.download('zh-hans')
stanza.download('zh-hant')
print('Done.')
