# writefile rfc_prototype/training/generate_dataset.py

import rebound
import numpy as np
import pandas as pd
import os
import csv
import multiprocessing
import sys
import time

from rfc_prototype.feature_extraction import compute_features, calculate_rho_crit_HW99, M_total, a_bin, P_bin, T_short, N_output_points, M_p_ref_for_features

# Check if running in a notebook or as a script
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Define the ranges for parameter sampling for two planets
mu_range = [0.1, 0.9] # Binary mass ratio μ = m_B / (m_A + m_B)
e_bin_range = [0.0, 0.7] # Binary eccentricity

# Planetary semimajor axis ratio ρ = a_p / a_bin (S-type range)
# For two planets, ensure a minimum separation
rho_min_global = 0.02 # Minimum rho for any planet
rho_max_global = 0.5  # Maximum rho for any planet

# Minimum number of mutual Hill radii separation for initial conditions
MIN_HILL_SEPARATION_FACTOR = 6.0

e_p_range = [0.0, 0.3] # Planetary eccentricity - This is now a *maximum* allowed, not a range for uniform sampling
inc_p_range_deg = [0.0, 40.0] # Planetary inclination relative to binary orbital plane in degrees
inc_p_range_rad = np.deg2rad(inc_p_range_deg) # Convert to radians for rebound
mean_anomaly_range_deg = [0.0, 360.0] # Initial planetary mean anomaly in degrees
mean_anomaly_range_rad = np.deg2rad(mean_anomaly_range_deg) # Convert to radians

# Simulation constants for long integration
N_periods_long = 1e4 # Integration time for long simulation, in binary periods (10^4)
T_long = N_periods_long * P_bin # Total integration time in rebound units

# Define thresholds for instability detection
MIN_DIST_FACTOR = 0.01 # Planet cannot get closer than 0.01*a_bin (physical collision threshold)
ESCAPE_DIST_FACTOR = 2.0 # Planet cannot go further than 2.0*a_bin from primary

def run_long_simulation_and_label_stability(initial_params):
    """
    Sets up and runs a long rebound simulation for a given set of parameters
    with TWO planets and determines the stability label.

    Args:
        initial_params (dict): A dictionary containing the initial parameters for the system.
                                Expected to contain 'mu', 'e_bin', 'rho_p1', 'e_p1', 'inc_p1',
                                'rho_p2', 'e_p2', 'inc_p2', 'mean_anomaly_p1', 'mean_anomaly_p2', 'id'.

    Returns:
        tuple: (int: 1 for stable, 0 for unstable, str: reason for instability or 'stable').
    """
    mu = initial_params['mu']
    e_bin = initial_params['e_bin']

    # Planet 1 parameters
    rho_p1 = initial_params['rho_p1']
    e_p1 = initial_params['e_p1']
    inc_p1_rad = initial_params['inc_p1'] # Already in radians
    mean_anomaly_p1_rad = initial_params['mean_anomaly_p1'] # Already in radians

    # Planet 2 parameters
    rho_p2 = initial_params['rho_p2']
    e_p2 = initial_params['e_p2']
    inc_p2_rad = initial_params['inc_p2'] # Already in radians
    mean_anomaly_p2_rad = initial_params['mean_anomaly_p2'] # Already in radians

    system_id = initial_params.get('id', 'N/A')

    m_B = mu * M_total
    m_A = M_total - m_B
    a_p1 = rho_p1 * a_bin
    a_p2 = rho_p2 * a_bin

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = 1.0

    sim.add(m=m_A)
    sim.add(m=m_B, primary=sim.particles[0], a=a_bin, e=e_bin, inc=0, Omega=0, omega=0, M=0)
    sim.add(m=M_p_ref_for_features, primary=sim.particles[0], a=a_p1, e=e_p1, inc=inc_p1_rad, Omega=0, omega=0, M=mean_anomaly_p1_rad)
    sim.add(m=M_p_ref_for_features, primary=sim.particles[0], a=a_p2, e=e_p2, inc=inc_p2_rad, Omega=0, omega=0, M=mean_anomaly_p2_rad)

    sim.move_to_com()
    sim.dt = sim.particles[1].P / 20.

    # Set up exit conditions for instability for both planets
    sim.exit_min_distance = MIN_DIST_FACTOR * a_bin

    stability_label = 0
    stability_reason = 'long_error' # Default to error until proven otherwise

    try:
        sim.integrate(T_long)
        stability_label = 1
        stability_reason = 'stable'
    except rebound.Escape as error:
        # sys.stderr.write(f"System {system_id} had an Escape during long integration.\n")
        stability_reason = 'escape'
    except rebound.Encounter as error:
        # sys.stderr.write(f"System {system_id} had an Encounter during long integration.\n")
        stability_reason = 'encounter'
    except Exception as e:
        sys.stderr.write(f"System {system_id} had an unexpected error during long integration: {e}\n")
        stability_reason = 'long_error'

    return stability_label, stability_reason

def worker_simulate_system(system_params):
    """
    Worker function to run both short and long simulations for a single system.
    Sets up the rebound.Simulation object and passes it to compute_features.
    """
    mu = system_params['mu']
    e_bin = system_params['e_bin']

    # Planet 1 parameters
    rho_p1 = system_params['rho_p1']
    e_p1 = system_params['e_p1']
    inc_p1_rad = system_params['inc_p1']
    mean_anomaly_p1_rad = system_params['mean_anomaly_p1']

    # Planet 2 parameters
    rho_p2 = system_params['rho_p2']
    e_p2 = system_params['e_p2']
    inc_p2_rad = system_params['inc_p2']
    mean_anomaly_p2_rad = system_params['mean_anomaly_p2']

    m_B = mu * M_total
    m_A = M_total - m_B
    a_p1 = rho_p1 * a_bin
    a_p2 = rho_p2 * a_bin

    # Create a fresh simulation object for feature extraction
    sim_for_features = rebound.Simulation()
    sim_for_features.integrator = "ias15"
    sim_for_features.G = 1.0

    sim_for_features.add(m=m_A)
    sim_for_features.add(m=m_B, primary=sim_for_features.particles[0], a=a_bin, e=e_bin, inc=0, Omega=0, omega=0, M=0)
    sim_for_features.add(m=M_p_ref_for_features, primary=sim_for_features.particles[0], a=a_p1, e=e_p1, inc=inc_p1_rad, Omega=0, omega=0, M=mean_anomaly_p1_rad)
    sim_for_features.add(m=M_p_ref_for_features, primary=sim_for_features.particles[0], a=a_p2, e=e_p2, inc=inc_p2_rad, Omega=0, omega=0, M=mean_anomaly_p2_rad)
    sim_for_features.move_to_com()
    sim_for_features.dt = sim_for_features.particles[1].P / 20.

    # First, run short simulation and extract features
    features = compute_features(sim_for_features, mu, e_bin)

    # Determine if short integration was successful (i.e., no NaNs indicating escape/encounter/error)
    # We check a few key features to decide this. If they are NaN, it implies instability during short sim.
    # The compute_features function returns a dict with NaNs if there was an issue.
    if np.isnan(features.get('megno_median_p1', np.nan)) or np.isnan(features.get('a_p_std_p1', np.nan)):
        features['stable'] = 0
        # Cannot infer a precise reason from compute_features directly, so we'll just say 'short_sim_unstable'
        # In a more refined design, compute_features could return a status flag.
        stability_reason_final = 'short_sim_unstable'
    else:
        # Then, run long simulation to determine stability label
        stability_label, long_reason = run_long_simulation_and_label_stability(system_params)
        features['stable'] = stability_label
        stability_reason_final = long_reason

    # Add the final stability reason to the features dict for tracking
    features['reason'] = stability_reason_final

    return features

def main():
    start_time = time.time()
    N_SYSTEMS = int(os.getenv('N_SYSTEMS', '100'))
    OUTPUT_FILENAME = f"s_type_stability_data_two_planets_{N_SYSTEMS}.csv"
    CHECKPOINT_INTERVAL = 50
    N_WORKERS = int(os.getenv('N_WORKERS', '11'))

    print(f"Generating {N_SYSTEMS} two-planet systems using {N_WORKERS} workers.")
    print(f"Output will be saved to {OUTPUT_FILENAME} with checkpointing every {CHECKPOINT_INTERVAL} systems.")

    list_of_params_for_workers = []
    num_sampled = 0

    reject_count_rho_p1_crit = 0
    reject_count_hill_sep = 0
    reject_count_ep1_too_high = 0
    reject_count_ep2_too_high = 0

    # The sampling loop for initial conditions
    while num_sampled < N_SYSTEMS:
        mu = np.random.uniform(*mu_range)
        e_bin = np.random.uniform(*e_bin_range)

        m_B = mu * M_total
        m_A = M_total - m_B

        rho_crit_HW99_val = calculate_rho_crit_HW99(mu, e_bin)

        rho_p1_min_effective = max(rho_min_global, 0.05) # Minimum periapsis > 0 for planet 1
        rho_p1_max_effective = min(rho_max_global, rho_crit_HW99_val * 0.7) # Apoapsis not too close to binary

        if rho_p1_max_effective <= rho_p1_min_effective:
            reject_count_rho_p1_crit += 1
            continue

        rho_p1 = np.random.uniform(rho_p1_min_effective, rho_p1_max_effective)
        a_p1 = rho_p1 * a_bin

        e_max_p1_peri = 1 - (0.05 * a_bin / a_p1)
        e_p1_upper_bound = min(e_p_range[1], e_max_p1_peri)

        if e_p1_upper_bound <= 0:
            reject_count_ep1_too_high += 1
            continue
        e_p1 = np.random.uniform(0, e_p1_upper_bound)

        apo_p1_current = a_p1 * (1 + e_p1)

        inc_p1 = np.random.uniform(*inc_p_range_rad)
        mean_anomaly_p1 = np.random.uniform(*mean_anomaly_range_rad)

        # Calculate required Hill radius separation from p1
        required_hill_sep = MIN_HILL_SEPARATION_FACTOR * a_p1 * (M_p_ref_for_features / (3 * m_A))**(1/3)

        a_p2_min_val = max(apo_p1_current, a_p1 + required_hill_sep)

        rho_p2_min_cand = a_p2_min_val / a_bin
        rho_p2_upper_bound = min(rho_max_global, rho_crit_HW99_val * 0.9)

        if rho_p2_upper_bound <= rho_p2_min_cand:
            reject_count_hill_sep += 1
            continue

        rho_p2 = np.random.uniform(rho_p2_min_cand, rho_p2_upper_bound)
        a_p2 = rho_p2 * a_bin

        e_max_p2_peri = 1 - (apo_p1_current / a_p2)
        e_max_p2_apo = (rho_crit_HW99_val * a_bin * 0.9 / a_p2) - 1

        e_p2_upper_bound = min(e_p_range[1], e_max_p2_peri, e_max_p2_apo)

        if e_p2_upper_bound <= 0:
            reject_count_ep2_too_high += 1
            continue
        e_p2 = np.random.uniform(0, e_p2_upper_bound)

        inc_p2 = np.random.uniform(*inc_p_range_rad)
        mean_anomaly_p2 = np.random.uniform(*mean_anomaly_range_rad)

        list_of_params_for_workers.append({
            'id': num_sampled,
            'mu': mu,
            'e_bin': e_bin,
            'rho_p1': rho_p1,
            'e_p1': e_p1,
            'inc_p1': inc_p1,
            'mean_anomaly_p1': mean_anomaly_p1,
            'rho_p2': rho_p2,
            'e_p2': e_p2,
            'inc_p2': inc_p2,
            'mean_anomaly_p2': mean_anomaly_p2
        })
        num_sampled += 1

    print(f"Rejected systems during initial condition sampling:")
    print(f"  - rho_p1 constrained by rho_crit_HW99 or min/max bounds: {reject_count_rho_p1_crit} systems")
    print(f"  - Planet 1 eccentricity too high (periapsis too close): {reject_count_ep1_too_high} systems")
    print(f"  - Insufficient Hill separation for p2 or p2 too close to p1/binary: {reject_count_hill_sep} systems")
    print(f"  - Planet 2 eccentricity too high (crossing p1/binary): {reject_count_ep2_too_high} systems")

    if not list_of_params_for_workers:
        print("No valid systems could be sampled with the given constraints. Please adjust parameters.")
        return

    # Prepare CSV file header by running one dummy simulation
    # Temporarily remove 'id' and 'reason' from dummy_features before getting fieldnames
    temp_dummy_params = list_of_params_for_workers[0].copy() # Make a copy to avoid modifying original
    dummy_features_raw = worker_simulate_system(temp_dummy_params) # This will return the 'reason' key

    # Ensure we get all feature keys including those that might be NaN from compute_features
    # We want keys from `compute_features` + 'stable'
    # To get a comprehensive list of fieldnames, we should create a dummy sim to get the feature dict keys.
    # Let's construct a dummy sim directly to ensure we get all expected feature keys from compute_features.

    # Create a dummy sim for header extraction
    dummy_mu, dummy_e_bin = 0.5, 0.1
    dummy_m_A = M_total * (1 - dummy_mu)
    dummy_m_B = M_total * dummy_mu
    dummy_a_p1 = 0.1 * a_bin # small enough for compute_features to potentially run
    dummy_e_p1 = 0.05
    dummy_inc_p1 = 0.0
    dummy_mean_anomaly_p1 = 0.0
    dummy_a_p2 = 0.2 * a_bin # larger than dummy_a_p1
    dummy_e_p2 = 0.05
    dummy_inc_p2 = 0.0
    dummy_mean_anomaly_p2 = 0.0

    dummy_sim = rebound.Simulation()
    dummy_sim.integrator = "ias15"
    dummy_sim.G = 1.0
    dummy_sim.add(m=dummy_m_A)
    dummy_sim.add(m=dummy_m_B, primary=dummy_sim.particles[0], a=a_bin, e=dummy_e_bin, inc=0, Omega=0, omega=0, M=0)
    dummy_sim.add(m=M_p_ref_for_features, primary=dummy_sim.particles[0], a=dummy_a_p1, e=dummy_e_p1, inc=dummy_inc_p1, Omega=0, omega=0, M=dummy_mean_anomaly_p1)
    dummy_sim.add(m=M_p_ref_for_features, primary=dummy_sim.particles[0], a=dummy_a_p2, e=dummy_e_p2, inc=dummy_inc_p2, Omega=0, omega=0, M=dummy_mean_anomaly_p2)
    dummy_sim.move_to_com()
    dummy_sim.dt = dummy_sim.particles[1].P / 20.

    # Get feature keys from a successful compute_features call
    dummy_features_for_header = compute_features(dummy_sim, dummy_mu, dummy_e_bin)
    fieldnames = list(dummy_features_for_header.keys()) + ['stable'] # Add the stability label to the fieldnames

    file_exists = os.path.isfile(OUTPUT_FILENAME)
    if not file_exists:
        with open(OUTPUT_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    escape_count = 0
    encounter_count = 0
    stable_count = 0
    other_unstable_count = 0
    short_sim_unstable_count = 0

    completed_systems = 0
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for i, result_features in enumerate(tqdm(
            pool.imap_unordered(worker_simulate_system, list_of_params_for_workers),
            total=N_SYSTEMS, desc="Simulating systems")
        ):
            if result_features['stable'] == 1:
                stable_count += 1
            elif result_features['reason'] == 'escape':
                escape_count += 1
            elif result_features['reason'] == 'encounter':
                encounter_count += 1
            elif result_features['reason'] == 'short_sim_unstable':
                short_sim_unstable_count += 1
            else:
                other_unstable_count += 1 # Includes 'long_error'

            # Remove the temporary 'reason' key before writing to CSV
            if 'reason' in result_features:
                del result_features['reason']

            with open(OUTPUT_FILENAME, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result_features)

            completed_systems += 1
            if completed_systems % CHECKPOINT_INTERVAL == 0:
                sys.stdout.write(f"\nCheckpoint: {completed_systems}/{N_SYSTEMS} systems processed and saved.\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nFinished generating {N_SYSTEMS} systems. Data saved to {OUTPUT_FILENAME}.")
    print(f"\nSummary of results:")
    print(f"  {escape_count} systems had an Escape during long integration.")
    print(f"  {encounter_count} systems had an Encounter during long integration.")
    print(f"  {short_sim_unstable_count} systems were unstable during short integration (e.g. escape/encounter/error).\n")
    print(f"  {stable_count} systems were stable.")
    print(f"  {other_unstable_count} systems were unstable due to other errors (long integration). ")
    print(f"Total elapsed time: {elapsed_time / 60:.2f} minutes.")

    if 'ipykernel' in sys.modules:
        if os.path.exists(OUTPUT_FILENAME):
            df_final = pd.read_csv(OUTPUT_FILENAME)
            print("\nFinal DataFrame head:")
            display(df_final.head())
            print(f"Total stable systems: {df_final['stable'].sum()} out of {len(df_final)}")
            print(f"Total unstable systems: {len(df_final) - df_final['stable'].sum()} out of {len(df_final)}")
        else:
            print(f"Output file {OUTPUT_FILENAME} not found. No DataFrame to display.")


if __name__ == '__main__':
    main()
