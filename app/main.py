from fastapi import FastAPI, HTTPException
import pandas as pd
import xgboost as xgb
import numpy as np
# Import the wrapper from your featurizer.py
from .featurizer import get_single_feature_vector

app = FastAPI(title="Materials Band Gap Predictor")

# 1. LOAD ASSETS
# These paths assume you are running uvicorn from the 'band-gap-service' folder
model = xgb.XGBRegressor()
model.load_model("app/model_assets/model_v2.json")
df_elements = pd.read_csv("app/model_assets/elemental_properties.csv", encoding='latin1')

# 2. CLEANING UTILITY
def clean_column_names(df):
    """
    Standardizes column names to avoid encoding mismatches (Å, •)
    between different operating systems and Python environments.
    """
    df.columns = [
        col.replace('Å', 'A')
           .replace('Ã…', 'A') # Common mojibake for Å
           .replace('Ã…', 'A') # Another variant
           .replace('•', '_')
           .replace('â€¢', '_') # Common mojibake for bullet
           .replace('â€¢', '_') # Another variant
        for col in df.columns
    ]
    return df

# Initial cleanup of the reference database
df_elements = clean_column_names(df_elements)

# Force all property columns to be numeric
for col in df_elements.columns[1:]:
    df_elements[col] = pd.to_numeric(df_elements[col], errors='coerce')

# 3. API ENDPOINT
@app.get("/predict")
def predict(formula: str):
    try:
        # Step A: Generate features from formula string
        input_df = get_single_feature_vector(formula, df_elements)

        # Step B: Clean the newly generated columns to match model training names
        input_df = clean_column_names(input_df)

        # Step C: Run Inference
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "input_formula": formula,
            "prediction": {
                "band_gap_ev": round(float(prediction), 4),
                "type": "Metal/Semiconductor/Insulator Check Pending"
            }
        }
    except Exception as e:
        # If anything fails (missing element, etc.), return a 400 error
        raise HTTPException(status_code=400, detail=str(e))

# For direct local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)