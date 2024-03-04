import pandas as pd
from fastapi import FastAPI
import joblib
from pandas import DataFrame
from pydantic import BaseModel


class FraudData(BaseModel):
    user_id: int
    signup_time: str
    purchase_time: str
    purchase_value: int
    device_id: str
    source: str
    browser: str
    sex: str
    age: int
    country: str


def preprocess_data(data: FraudData, bin_encoder):
    data_dict = data.model_dump()
    data = DataFrame([data_dict])
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    data['time_difference'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds().astype(int)
    data['day_purchase'] = data['purchase_time'].dt.day
    data['hour_purchase'] = data['purchase_time'].dt.hour
    data['minute_purchase'] = data['purchase_time'].dt.minute
    data['second_purchase'] = data['purchase_time'].dt.second
    columns_drop = ['signup_time', 'purchase_time']
    data = data.drop(columns_drop, axis=1)
    data_encoded = bin_encoder.transform(data)
    return data_encoded


fastAPI_model = FastAPI(title='ML fraud detection model',
                        description='Deploy the ML model to production',
                        version='1.0')
@fastAPI_model.get("/")
async def root():
    return {"message": "Welcome to the E-Commerce Fraud ML API model"}


@fastAPI_model.post("/predict/")
def predict(data: FraudData):
    # Load the trained model
    model = joblib.load('RFC_model.joblib')
    # load the categorical encoder from the model
    binary_encoder = joblib.load('binary_encoder.joblib')
    processed_data = preprocess_data(data, binary_encoder)
    # Make predictions using the loaded model
    prediction = model.predict(processed_data)
    original_data = pd.DataFrame([data.model_dump()])
    original_data['class'] = prediction
    dict_data = original_data.to_dict(orient='records')[0]
    return dict_data