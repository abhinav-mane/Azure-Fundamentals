from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/predict")
def predict():
    responses = ["sonam weds abhinav", "abhinav weds sonam", "he weds she","she weds he"]
    return {"prediction": random.choice(responses)}
