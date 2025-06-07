
from datetime import datetime
from fastapi import FastAPI
import logging
import pandas as pd
from pathlib import Path
import torch
import uvicorn
from src.model_demo.utils import LinearRegressionModel, PredictionFeatures, infer_model

DATA_DIR =Path("data/model_demo")
MODEL_DIR = Path("models/model_demo")
MODEL_FNAME = "demo_model_weights.pth"
## Logger setup
logger = logging.getLogger(__name__) # or custom name insead of __name__
logger.setLevel(logging.DEBUG)  

# DEBUG (10) >INFO (20) > WARNING (30) > ERROR (40) > CRITICAL (50)
# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console shows INFO and above

# Create a formatter
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
    )
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)

# Initialize FastAPI app
app = FastAPI(title="Demo model API", description="API for simple linear model prediction")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using {device} device")
logger.info(f"Running at: {Path.cwd()}")


# create an instance of the same model first
model = LinearRegressionModel(2,1)

# Load the trained model weights, weights_only=True as a best practice.
model.load_state_dict(torch.load(MODEL_DIR / MODEL_FNAME, weights_only=True))
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

    with open(DATA_DIR / 'predictions.txt', 'a') as f:
        f.write(f"{datetime.now()}\nInput:\n{input_df}\nPrediction:\n{outputs}\n\n")

    logger.info(f"Input: {input_df}, Prediction: {outputs}")

    return {
        "Model prediction": outputs
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    


# https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/
