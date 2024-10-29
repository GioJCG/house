from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware


origins = ["*"]

app = FastAPI(title = 'Housing Price Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

model = load(pathlib.Path('model/housing-v1.joblib'))

class InputData(BaseModel):
    longitude: float=-122.23
    latitude: float=37.88
    housing_median_age: float=41
    total_rooms: int=880
    total_bedrooms: int=129
    population: int=322
    households: int=126
    median_income: float=8.3254

class OutputData(BaseModel):
    price: float

@app.post('/score', response_model=OutputData)
def predict(data: InputData):
    
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1, -1)
    
    result = model.predict(model_input)[0]

    return OutputData(price=result)
