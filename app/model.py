import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Pass" if prediction == 1 else "Fail"
