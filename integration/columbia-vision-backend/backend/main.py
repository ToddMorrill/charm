"""This module serves TA1.* systems running on AWS.

Examples:
    $ uvicorn main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel
# import torch
from ccu import CCU
from pathlib import Path


import logging
from PIL import Image
from io import BytesIO
import json
# for face detection
from facenet_pytorch import MTCNN

# clip
import clip

import torch
import numpy as np

similarity_scale = 150.0

# helper functions
# TODO: add these to a utils file
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

app = FastAPI()


class Item(BaseModel):
    img_string: str

# TODO: set log level from environment variable
logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)
logging.info('Starting columbia-vision-backend')

# TODO: need to implement some sort of readiness check to prevent requests before model is loaded

# initializing clip model
# device = "cuda:7" if torch.cuda.is_available() else "cpu"
logging.info('Loading CLIP...')
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

model, preprocess = clip.load("ViT-B/16")
# model, preprocess = clip.load("RN50x16")
model.to(device).eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size


print("----- CLIP model info -----")
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
print("---------------------------")

def zeroshot_emotion_classifier(emotion_clip_prompts_path):
	clip_prompts = read_json(emotion_clip_prompts_path)

	with torch.no_grad():
		zeroshot_weights = []
		for classname in clip_prompts:
			texts = []

			for t in clip_prompts[classname]:
				# print(f"{classname} --> {t}")
				texts.append(t)
			# print("--")
			texts = clip.tokenize(texts, truncate=True).to(device) #tokenize
			class_embeddings = model.encode_text(texts) #embed with text encoder
			class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
			class_embedding = class_embeddings.mean(dim=0)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)

		zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
	return zeroshot_weights

def classify_image(image, zeroshot_weights, emotion_classes, similarity_scale=150.0):
    """
    emotion_classes = list of emotions. Idx of list gives emotion name
    """
    with torch.no_grad():
        images = preprocess(image).unsqueeze(0).to(device)
        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # print("image_features shape= ", image_features.shape)
        # print("zeroshot_weights shape= ", zeroshot_weights.shape)
        # print("image_features = ", image_features)
        # print("zeroshot_weights = ", zeroshot_weights)

        logits = image_features @ zeroshot_weights
        # print("logits= ", logits)
        logits = logits.cpu()
        preds =  torch.argmax(logits, dim =1)
        # 0th index coz only one image
        predicted_emotion = emotion_classes[preds[0]]

        logits = logits[0].float()
        # print("logits= ", logits)
        probs = torch.nn.functional.softmax(logits*similarity_scale, dim=0)
        # print("probs= ", probs)
        predicted_emotion_prob = probs[preds[0]].item()

    return predicted_emotion, logits, predicted_emotion_prob

# initializing face detection
logging.info('Loading mtcnn...')
mtcnn = MTCNN(select_largest=False) # for face detection 

emotion_clip_prompts_path = "emotion_clip_prompts.json"
emotion_clip_prompts = read_json(emotion_clip_prompts_path)
zeroshot_classifier_weights = zeroshot_emotion_classifier(emotion_clip_prompts_path)
# print("zeroshot_classifier_weights shape=", zeroshot_classifier_weights.shape)

logging.info('Loaded: all models initialized')


def classify_msg(img_byte_string):
    logging.info('Got image to classify')
    # # debug
    # logging.debug(f"img_byte_string: {img_byte_string}")
    # return {"got string": img_byte_string}
    # # debug

    image_bytes = CCU.base64string_to_binary(img_byte_string)
    # with open('image.jpg', 'wb') as file:
    #     file.write(image_bytes) 
    # print(f'Got a frame with {len(image_bytes)} data.')
    stream = BytesIO(image_bytes)
    frame = Image.open(stream).convert("RGB") # WORKS!!

    boxes, probs = mtcnn.detect([frame])
    # highest prob is None = No face found
    if probs[0][0] != None:
        # print(f"No face in frame, skipping this frame")
        predicted_emotion, logits, predicted_emotion_prob = classify_image(frame, zeroshot_classifier_weights, list(emotion_clip_prompts.keys()))
        # print(f"predicted emotion for frame with prob = {predicted_emotion} {predicted_emotion_prob} ")
        frame_emotion = predicted_emotion
        frame_emotion_probability = predicted_emotion_prob
        
        prediction_json = {
            "frame_emotion": frame_emotion,
            "frame_emotion_probability": frame_emotion_probability         
        }
        return prediction_json
    
    else:
        return {
            "error": "no face found"
        }

@app.post("/predict")
async def predict(item: Item):
    # logging.info("item.img_string: ", item.img_string)
    predictions = classify_msg(item.img_string)
    return predictions
