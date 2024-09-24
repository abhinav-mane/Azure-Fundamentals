from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc
import pandas as pd
import joblib

from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core import Run
import json
import os
import numpy as np


def init():
    global model
    # Load the model
    model_path = Model.get_model_path('bankdatamodel')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Parse the incoming JSON request
        data = json.loads(raw_data)
        
        # Extract the 'data' field assuming it's passed as a key in the JSON body
        input_data = data.get("data", None)

        # Convert to a NumPy array for the model
        if input_data is not None:
            input_data = np.array(input_data)

            # Make a prediction
            prediction = model.predict(input_data)

            # Return the result as a JSON response
            return json.dumps({"result": prediction.tolist()})
        else:
            return json.dumps({"error": "Invalid input format: 'data' key not found."})
    
    except Exception as e:
        # Return the error message if something goes wrong
        return json.dumps({"error": str(e)})

