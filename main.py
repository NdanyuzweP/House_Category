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




@app.post("/retrain-model")
async def retrain_model(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = f"datasets/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Trigger retraining script
        subprocess.run(["python", "retrain_script.py", file_path], check=True)
        return {"message": "Retraining started successfully"}
    except Exception as e:
        return {"error": str(e)}
