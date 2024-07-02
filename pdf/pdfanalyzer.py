from transformers import pipeline
import os

pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

pipe(
    os.path.join(os.curdir, 'example.pdf')
)