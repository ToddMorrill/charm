"""This module serves TA1.* systems running on AWS.

Examples:
    $ uvicorn main:app --reload
"""
import logging

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

app = FastAPI()


class Batch(BaseModel):
    # {'text': ['string1', 'string2'],
    #  'audio': 'base64 encoded string'}
    text: list[str]


# TODO: set log level from environment variable
logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)
logging.info('Starting columbia-communication-change-backend')

# TODO: need to implement some sort of readiness check to prevent requests before model is loaded
model_name = "liam168/c2-roberta-base-finetuned-dianping-chinese"
class_num = 2
ts_texts = ["我喜欢下雨。", "我讨厌他."]
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=class_num)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# TODO: create an environment variable to control which GPU is used
if torch.cuda.is_available():
    logging.debug('CUDA available - using the GPU.')
classifier = pipeline('sentiment-analysis',
                      model=model,
                      tokenizer=tokenizer,
                      device=0)


@app.post("/predict")
def predict(batch: Batch):
    # TODO: more validation of inputs
    predictions = classifier(batch.text)
    return predictions
