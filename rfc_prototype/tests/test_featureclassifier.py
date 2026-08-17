# writefile rfc_prototype/tests/test_featureclassifier.py

import pytest
import joblib
import os
import rebound
import numpy as np
import pandas as pd

# Ensure rebound is installed for the test environment
try:
    import rebound
except ImportError:
    print("rebound not found, installing...")
    os.system('pip install rebound')
    import rebound

# Define the path to the packaged model relative to the test file
# Assuming test_featureclassifier.py is in rfc_prototype/tests/
# and the model is in rfc_prototype/rfc_prototype/models/
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', # Go up to rfc_prototype/
    'rfc_prototype',
    'models',
    'rfc_stype_2planet.joblib'
)

# --- MOCK compute_features FUNCTION ---
# IMPORTANT: In your actual project, you will import the real compute_features from
# rfc_prototype.feature_extraction. This mock is for testing purposes only.
# It calculates some features directly from the rebound simulation and assigns
# plausible dummy values for features that would normally come from longer integrations.

def mock_compute_features(sim: rebound.Simulation) -> dict:
    """Mock function to extract features from a rebound simulation.
    This should be replaced by your actual rfc_prototype.feature_extraction.compute_features.
    """
    features = {}

    # Assuming particles: 0=star_A, 1=star_B, 2=planet_1, 3=planet_2
    # Binary parameters
    m_A = sim.particles[0].m
    m_B = sim.particles[1].m
    a_bin = sim.particles[1].a # Assuming binary orbit is defined by relative orbit of B around A
    e_bin = sim.particles[1].e

    features['mu'] = m_B / (m_A + m_B)
    features['e_bin'] = e_bin

    # Planet 1 parameters
    p1 = sim.particles[2]
    features['rho_p1'] = p1.a / a_bin
    features['inc_p_deg_p1'] = np.degrees(p1.inc)

    # Planet 2 parameters
    p2 = sim.particles[3]
    features['rho_p2'] = p2.a / a_bin
    features['inc_p_deg_p2'] = np.degrees(p2.inc)

    # Planet-planet interactions
    features['period_ratio_p2_p1'] = p2.P / p1.P
    features['initial_separation_au'] = p2.a - p1.a
    features['init_mutual_inc_deg'] = np.degrees(abs(p1.inc - p2.inc))

    # MMR flags
    features['mmr_flag_2_1'] = 1 if 1.9 <= features['period_ratio_p2_p1'] <= 2.1 else 0
    features['mmr_flag_3_2'] = 1 if 1.4 <= features['period_ratio_p2_p1'] <= 1.6 else 0

    # Placeholder values for features that require integration/complex calculation
    # These values are chosen to be indicative of stable/unstable for the specific test cases.
    # YOU MUST REPLACE THESE WITH ACTUAL CALCULATIONS FROM YOUR feature_extraction.py
    features['e_p_free_p1'] = 0.01 # Dummy: low for stable, higher for unstable
    features['e_p_forced_p1'] = 0.05 # Dummy
    features['megno_median_p1'] = 2.05 # Dummy: near 2 for stable, >2 for unstable
    features['megno_std_p1'] = 0.01 # Dummy: low for stable, higher for unstable
    features['a_p_std_p1'] = 0.001 # Dummy: low for stable, higher for unstable

    features['e_p_free_p2'] = 0.02 # Dummy
    features['e_p_forced_p2'] = 0.07 # Dummy
    features['megno_median_p2'] = 2.08 # Dummy
    features['megno_std_p2'] = 0.02 # Dummy
    features['a_p_std_p2'] = 0.002 # Dummy

    # Ensure all 21 features are present, fill with reasonable defaults if any are missing for some reason
    # This list should match the `feature_columns` in your training script.
    expected_feature_columns = [
        'mu', 'e_bin',
        'rho_p1', 'e_p_free_p1', 'e_p_forced_p1', 'inc_p_deg_p1', 'megno_median_p1', 'megno_std_p1', 'a_p_std_p1',
        'rho_p2', 'e_p_free_p2', 'e_p_forced_p2', 'inc_p_deg_p2', 'megno_median_p2', 'megno_std_p2', 'a_p_std_p2',
        'period_ratio_p2_p1', 'initial_separation_au', 'init_mutual_inc_deg', 'mmr_flag_2_1', 'mmr_flag_3_2'
    ]

    # For the mock, we'll assume the simple features above are computed correctly,
    # and the complex ones get values set below that are manually adjusted for stable/unstable scenarios.

    # This part should be handled by the actual compute_features, which ensures all 21 features are computed.
    # The values for e_p_free, e_p_forced, megno_median/std, a_p_std will be critical.
    # For this mock, we'll set them to typical 'stable' values as a baseline.
    for col in expected_feature_columns:
        if col not in features:
            features[col] = 0.0 # Default value, should be replaced by real computation

    return features

# --- PYTEST FIXTURES ---

@pytest.fixture(scope='module')
def trained_model():
    """Loads the pre-trained RFC model."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found at {MODEL_PATH}. Run training script first.")
    model = joblib.load(MODEL_PATH)
    return model

@pytest.fixture
def stable_sim():
    """Creates a rebound simulation expected to be stable."""
    sim = rebound.Simulation()
    sim.units = ('AU', 'M_sun', 'yr')
    sim.add(m=0.5, hash="star_A")
    sim.add(m=0.5, a=1.0, e=0.01, hash="star_B") # Binary a=1 AU, low e
    # Planet 1: well inside the binary, low ecc, low inc, well separated from p2
    sim.add(m=1e-5, a=0.1, e=0.01, inc=np.radians(1), hash="planet_1")
    # Planet 2: further out, low ecc, low inc, well separated from p1
    sim.add(m=1e-5, a=0.2, e=0.01, inc=np.radians(1), hash="planet_2")

    # Modify mock features for stable scenario (these should come from actual compute_features)
    sim.mock_feature_values = {
        'e_p_free_p1': 0.01, 'e_p_forced_p1': 0.02, 'megno_median_p1': 2.001, 'megno_std_p1': 0.001, 'a_p_std_p1': 0.0001,
        'e_p_free_p2': 0.01, 'e_p_forced_p2': 0.03, 'megno_median_p2': 2.002, 'megno_std_p2': 0.002, 'a_p_std_p2': 0.0002,
        'period_ratio_p2_p1': 2.828, # sqrt(0.2^3/0.1^3) is approx 2.828
        'mmr_flag_2_1': 0, 'mmr_flag_3_2': 0 # Not in resonance
    }
    return sim

@pytest.fixture
def unstable_sim():
    """Creates a rebound simulation expected to be unstable (e.g., in a strong resonance)."""
    sim = rebound.Simulation()
    sim.units = ('AU', 'M_sun', 'yr')
    sim.add(m=0.5, hash="star_A")
    sim.add(m=0.5, a=1.0, e=0.1, hash="star_B") # Binary a=1 AU, higher e
    # Planet 1: closer to binary, higher ecc, higher inc
    sim.add(m=1e-5, a=0.15, e=0.1, inc=np.radians(10), hash="planet_1")
    # Planet 2: in 2:1 resonance with p1, higher ecc, higher inc
    # For 2:1 resonance: a2 = a1 * (2/1)^(2/3) approx a1 * 1.587
    sim.add(m=1e-5, a=0.238, e=0.1, inc=np.radians(10), hash="planet_2")

    # Modify mock features for unstable scenario (these should come from actual compute_features)
    sim.mock_feature_values = {
        'e_p_free_p1': 0.1, 'e_p_forced_p1': 0.2, 'megno_median_p1': 2.5, 'megno_std_p1': 0.1, 'a_p_std_p1': 0.01,
        'e_p_free_p2': 0.1, 'e_p_forced_p2': 0.25, 'megno_median_p2': 2.8, 'megno_std_p2': 0.15, 'a_p_std_p2': 0.015,
        'period_ratio_p2_p1': 1.587, # This is approx (2/1)^(2/3) indicating potential 2:1 resonance
        'mmr_flag_2_1': 1, 'mmr_flag_3_2': 0 # In 2:1 resonance
    }
    return sim

# --- TEST FUNCTIONS ---

def test_model_loads(trained_model):
    """Tests that the model fixture successfully loads the joblib file."""
    assert trained_model is not None
    assert hasattr(trained_model, 'predict')

def test_stable_prediction(trained_model, stable_sim):
    """Tests that the model predicts 'stable' for a known stable configuration."""
    # Augment features with mock values for the complex ones
    features_dict = mock_compute_features(stable_sim)
    features_dict.update(stable_sim.mock_feature_values) # Apply scenario-specific mock values

    feature_vector = pd.DataFrame([features_dict])

    # Ensure feature columns are in the correct order as per training
    expected_feature_columns = [
        'mu', 'e_bin',
        'rho_p1', 'e_p_free_p1', 'e_p_forced_p1', 'inc_p_deg_p1', 'megno_median_p1', 'megno_std_p1', 'a_p_std_p1',
        'rho_p2', 'e_p_free_p2', 'e_p_forced_p2', 'inc_p_deg_p2', 'megno_median_p2', 'megno_std_p2', 'a_p_std_p2',
        'period_ratio_p2_p1', 'initial_separation_au', 'init_mutual_inc_deg', 'mmr_flag_2_1', 'mmr_flag_3_2'
    ]
    feature_vector = feature_vector[expected_feature_columns]

    prediction = trained_model.predict(feature_vector)[0]
    # A stable system should predict 1 (stable)
    assert prediction == 1, f"Expected stable (1) but got {prediction} for stable_sim"
    print(f"Stable sim predicted: {prediction}")

def test_unstable_prediction(trained_model, unstable_sim):
    """Tests that the model predicts 'unstable' for a known unstable configuration."""
    # Augment features with mock values for the complex ones
    features_dict = mock_compute_features(unstable_sim)
    features_dict.update(unstable_sim.mock_feature_values) # Apply scenario-specific mock values

    feature_vector = pd.DataFrame([features_dict])

    # Ensure feature columns are in the correct order as per training
    expected_feature_columns = [
        'mu', 'e_bin',
        'rho_p1', 'e_p_free_p1', 'e_p_forced_p1', 'inc_p_deg_p1', 'megno_median_p1', 'megno_std_p1', 'a_p_std_p1',
        'rho_p2', 'e_p_free_p2', 'e_p_forced_p2', 'inc_p_deg_p2', 'megno_median_p2', 'megno_std_p2', 'a_p_std_p2',
        'period_ratio_p2_p1', 'initial_separation_au', 'init_mutual_inc_deg', 'mmr_flag_2_1', 'mmr_flag_3_2'
    ]
    feature_vector = feature_vector[expected_feature_columns]

    prediction = trained_model.predict(feature_vector)[0]
    # An unstable system should predict 0 (unstable)
    assert prediction == 0, f"Expected unstable (0) but got {prediction} for unstable_sim"
    print(f"Unstable sim predicted: {prediction}")
