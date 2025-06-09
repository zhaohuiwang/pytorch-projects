

import logging
import pandas as pd
import torch
import uvicorn

from datetime import datetime
from fastapi import FastAPI
from pathlib import Path
from src.model_demo.config import MetadataConfigSchema
from src.model_demo.utils import LinearRegressionModel, PredictionFeatures, infer_model, setup_logger, device

data_dir = "data/model_demo"
model_dir = "models/model_demo"
model_fname = "demo_model_weights.pth"


config = MetadataConfigSchema()
data_dir = config.data.data_dir
data_fname = config.data.data_fname
model_dir = config.data.model_dir
model_fname = config.data.model_fname


## Logger setup
logger = setup_logger(logger_name=__name__, log_file=f'{data_dir}/api_logfile.log')

# Initialize FastAPI app
app = FastAPI(title="Demo model API", description="API for simple linear model prediction")

logger.info(f"Using {device} device")
logger.info(f"Running at: {Path.cwd()}")


# create an instance of the same model first
model = LinearRegressionModel(2,1)

# Load the trained model weights, weights_only=True as a best practice.
model.load_state_dict(torch.load(Path(model_dir) / model_fname, weights_only=True))
model.to(device)
model.eval()  # Set to evaluate mode

# API Root endpoint
@app.get("/")
async def index():
    return {"message": "Welcome to the model demo API. Use the /predict feature to predict your outcome."}

# Prediction endpoint
@app.post("/predict")
async def predict(features: PredictionFeatures):
# defined an asynchronous function named prediction - allowing other tasks to run while it waits for I/O-bound operations

    # Create input DataFrame for prediction
    input_df = pd.DataFrame([{
        "X_1": features.feature_X_1,
        "X_2": features.feature_X_2
    }])

    # Convert DataFrame to NumPy array
    np_array = input_df.to_numpy()  # or df.values

    # Convert NumPy array to PyTorch tensor
    inputs = torch.tensor(np_array)

    # model inference
    outputs = infer_model(model, inputs)

    with open(Path(data_dir) / 'predictions.txt', 'a') as f:
        f.write(f"{datetime.now()}\nInput:\n{input_df}\nPrediction:\n{outputs}\n\n")

    logger.info(f"Input: {input_df}, Prediction: {outputs}")

    return {
        "Model prediction": outputs
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
        


# https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/
