import joblib
import rebound
import numpy as np
import pandas as pd
from importlib.resources import files
from .feature_extraction import compute_features

class FeatureClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            #model_path = files("your_package_name.models").joinpath("rfc_stype_2planet.joblib")
            model_path = files("rfc_prototype.models").joinpath("rfc_stype_2planet.joblib")
        self.model = joblib.load(model_path)

    def predict_stable(self, sim: rebound.Simulation, mu: float, e_bin: float) -> float:
        if sim.N - 1 > 3:  # star + binary companion + up to 2 planets
            raise ValueError("This model supports at most 2 planets around the primary.")

        # Extract features from the simulation
        features = compute_features(sim, mu, e_bin)

        # Convert features dictionary to a format suitable for the model
        feature_df = pd.DataFrame([features])

        #X = np.array(features).reshape(1, -1)
        #return self.model.predict_proba(X)[0, 1]

        prediction_probability = 0

        if self.model:
            # Ensure feature order and preprocessing match training data
            prediction_probability = self.model.predict_proba(feature_df[self.feature_columns])[:, 1][0]

        return prediction_probability
