from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import pandas as pd
from io import StringIO

# Load the model
model = tf.keras.models.load_model("models/house_model.h5")
scaler = joblib.load('./models/scaler.pkl')

# Initializing FastAPI app
app = FastAPI()


# Defining the input data model
class HouseData(BaseModel):
    squareMeters: float
    numberOfRooms: int
    hasYard: int
    hasPool: int
    floors: int
    numPrevOwners: int
    made: float
    isNewBuilt: int
    hasStormProtector: int
    basement: float
    attic: float
    garage: float
    hasStorageRoom: int
    hasGuestRoom: int
    price: float

# Define prediction endpoint
import pandas as pd

@app.post("/predict")
def predict(data: HouseData):
    # Convert input data into a DataFrame
    input_dict = {
        "squareMeters": [data.squareMeters],
        "numberOfRooms": [data.numberOfRooms],
        "hasYard": [data.hasYard],
        "hasPool": [data.hasPool],
        "floors": [data.floors],
        "numPrevOwners": [data.numPrevOwners],
        "made": [data.made],
        "isNewBuilt": [data.isNewBuilt],
        "hasStormProtector": [data.hasStormProtector],
        "basement": [data.basement],
        "attic": [data.attic],
        "garage": [data.garage],
        "hasStorageRoom": [data.hasStorageRoom],
        "hasGuestRoom": [data.hasGuestRoom],
        "price": [data.price]
    }
    input_df = pd.DataFrame(input_dict)
    
    # Scale the appropriate columns using the scaler
    columns_to_scale = ["squareMeters", "made", "basement", "attic", "garage", "price"]
    input_df[columns_to_scale] = scaler.fit_transform(input_df[columns_to_scale])
        
    # Convert the DataFrame to a NumPy array for prediction
    input_data = input_df.to_numpy()
    
    # Make prediction
    prediction = model.predict(input_data)
    print(prediction)
    
    # Convert prediction to class label (Luxury or Basic)
    category = "Luxury" if prediction[0][0] > 0.5 else "Basic"
    
    return {"category": category, "probability": float(prediction[0][0])}


# Endpoint for retraining the model with new data
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
    data_str = contents.decode("utf-8")
    data = pd.read_csv(StringIO(data_str))
    
    # Ensure the uploaded data has the right format
    expected_columns = [
        "squareMeters", "numberOfRooms", "hasYard", "hasPool", "floors", 
        "numPrevOwners", "made", "isNewBuilt", "hasStormProtector", 
        "basement", "attic", "garage", "hasStorageRoom", "hasGuestRoom", "price"
    ]
    
    if not all(col in data.columns for col in expected_columns):
        missing_columns = [col for col in expected_columns if col not in data.columns]
        return {"error": f"Invalid data format. Missing columns: {missing_columns}"}
    
    # Scale the data including 'price' to match scaler's expectations
    scaled_data = data.copy()
    columns_to_scale = ["squareMeters", "made", "basement", "attic", "garage", "price"]
    scaled_data[columns_to_scale] = scaler.transform(scaled_data[columns_to_scale])
    
    # Separate features (X) and target (y)
    X = scaled_data.drop("price", axis=1)  # Features
    y = scaled_data["price"]  # Target
    
    # Retrain the model
    model.fit(X.to_numpy(), y.to_numpy(), epochs=10, batch_size=32, verbose=1)
    
    # Save the retrained model
    model.save("models/house_model.h5")

    # Return success message
    return {"message": "Model retrained successfully!"}
