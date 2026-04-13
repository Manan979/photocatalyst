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
        # 1. Get the raw features
        input_df = get_single_feature_vector(formula, df_elements)

        # 2. THE HAMMER: Force column names to match the model's training names
        # This bypasses all encoding issues (Å, Ã…, etc.)
        model_features = model.get_booster().feature_names
        
        if len(input_df.columns) == len(model_features):
            input_df.columns = model_features
        else:
            # If the count is different, we have a bigger featurizer issue
            raise Exception(f"Feature count mismatch. Model needs {len(model_features)}, got {len(input_df.columns)}")

        # 3. Predict
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "input_formula": formula,
            "prediction": {
                "band_gap_ev": round(float(prediction), 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# For direct local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
