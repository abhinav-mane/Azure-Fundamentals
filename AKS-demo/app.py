from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/predict")
def predict():
    responses = ["yes", "no", "dont know"]
    return {"prediction": random.choice(responses)}
