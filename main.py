from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np

# ✅ FastAPI app object (THIS IS CRITICAL)
app = FastAPI(title="Network Intrusion Detection System")

# Templates & static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model, scaler, encoder = pickle.load(open("model.pkl", "rb"))

# Input schema
class IntrusionInput(BaseModel):
    duration: float
    src_bytes: float
    dst_bytes: float

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: IntrusionInput):
    input_data = np.array([[data.duration, data.src_bytes, data.dst_bytes]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    label = encoder.inverse_transform(prediction)

    return {"prediction": label[0]}
