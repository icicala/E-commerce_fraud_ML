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
    ip_address: int
    country: str


def preprocess_data(data: FraudData, bin_encoder):
    data_dict = data.model_dump()
    data = DataFrame([data_dict])
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    data['time_difference'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds().astype(int)
    data['signup_day'] = data['signup_time'].dt.day
    data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    data['signup_time_seconds'] = data['signup_time'].dt.hour * 3600 + data['signup_time'].dt.minute * 60 + data[
        'signup_time'].dt.second
    data['purchase_week'] = data['purchase_time'].dt.isocalendar().week
    data['purchase_day'] = data['purchase_time'].dt.day
    data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    data['purchase_time_seconds'] = data['purchase_time'].dt.hour * 3600 + data['purchase_time'].dt.minute * 60 + data[
        'purchase_time'].dt.second
    data['number_user_per_device'] = data.groupby('device_id')['user_id'].transform('count')
    data['number_user_per_country'] = data.groupby('country')['user_id'].transform('count')
    columns_drop = ['signup_time', 'purchase_time', 'device_id', 'ip_address', 'country', 'source', 'browser', 'sex',
               'user_id']
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