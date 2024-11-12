from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

# Load the trained model
with open('diabetes_model_improved.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def read_root():
    return FileResponse("index.html")
@app.get("/style.css")
def get_css():
    return FileResponse("style.css")

@app.get("/script.js")
def get_js():
    return FileResponse("script.js")

@app.post("/predict")
def predict(input_data: DiabetesInput):
    features = [
        input_data.Pregnancies, input_data.Glucose, input_data.BloodPressure,
        input_data.SkinThickness, input_data.Insulin, input_data.BMI,
        input_data.DiabetesPedigreeFunction, input_data.Age
    ]
    prediction = model.predict([features])
    result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
    return {"diabetes_result": result}
