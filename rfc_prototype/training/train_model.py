# writefile rfc_prototype/training/rfc_prototype/training/train_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Assume this function is provided by rfc_prototype.feature_extraction
# For the purpose of this script, we'll define a placeholder if not present, but in your actual project,
# it should be imported.
from rfc_prototype.feature_extraction import compute_features

def compute_features(raw_data_row):
    """Placeholder for feature extraction. In your project, this would extract
    the ~20 relevant features from raw simulation data for a two-planet system.
    This function should return a dictionary of features.
    """
    # Example: if raw_data_row is a Series, just return existing columns as features
    # In a real scenario, this would involve more complex calculations.
    features = raw_data_row.drop(columns=['stable'], errors='ignore').to_dict()
    return features

def train_and_save_model(data_path, model_save_path):
    # Load the CSV data
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    # Define features (X) and target (y)
    # These feature names must match the columns in your CSV after feature extraction
    feature_columns = [
        'mu', 'e_bin',
        'rho_p1', 'e_p_free_p1', 'e_p_forced_p1', 'inc_p_deg_p1', 'megno_median_p1', 'megno_std_p1', 'a_p_std_p1',
        'rho_p2', 'e_p_free_p2', 'e_p_forced_p2', 'inc_p_deg_p2', 'megno_median_p2', 'megno_std_p2', 'a_p_std_p2',
        'period_ratio_p2_p1', 'initial_separation_au', 'init_mutual_inc_deg', 'mmr_flag_2_1', 'mmr_flag_3_2'
    ]

    X = df[feature_columns]
    y = df['stable']

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nData split into training and testing sets.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Initialize and train the Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    print("\nRandomForestClassifier trained successfully.")

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = np.nan

    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}" if not np.isnan(auc) else "AUC: N/A")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the fitted model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf_model, model_save_path)
    print(f"\nFitted model saved to: {model_save_path}")

if __name__ == '__main__':
    # Define paths relative to the project root for standalone execution
    project_root = os.path.dirname(os.path.abspath(__file__)).split('rfc_prototype')[0]
    # Adjust path if rfc_prototype/training/rfc_prototype/training is nested
    if project_root.endswith('rfc_prototype/'):
        project_root = project_root[:-len('rfc_prototype/')]

    data_filepath = os.path.join(project_root, 'rfc_prototype', 'data', 's_type_stability_data_two_planets_10000.csv')
    model_filepath = os.path.join(project_root, 'rfc_prototype', 'rfc_prototype', 'models', 'rfc_stype_2planet.joblib')

    train_and_save_model(data_filepath, model_filepath)
