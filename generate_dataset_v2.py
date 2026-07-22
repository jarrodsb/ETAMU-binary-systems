# generate_dataset_v2.py

import rebound
import numpy as np
import pandas as pd
import os
import csv
import multiprocessing
import sys
import time

# Check if running in a notebook or as a script
# This helps select the correct tqdm import for better display
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Define the ranges for parameter sampling
mu_range = [0.1, 0.9] # Binary mass ratio μ = m_B / (m_A + m_B)
e_bin_range = [0.0, 0.7] # Binary eccentricity
rho_range = [0.02, 0.5] # Planetary semimajor axis ratio ρ = a_p / a_bin (S-type range)
e_p_range = [0.0, 0.3] # Planetary eccentricity
inc_p_range_deg = [0.0, 40.0] # Planetary inclination relative to binary orbital plane in degrees
inc_p_range_rad = np.deg2rad(inc_p_range_deg) # Convert to radians for rebound
mean_anomaly_range_deg = [0.0, 360.0] # Initial planetary mean anomaly in degrees
mean_anomaly_range_rad = np.deg2rad(mean_anomaly_range_deg) # Convert to radians

# Total binary mass and binary semimajor axis (rebound units G=1)
M_total = 1.0 # total binary mass in solar masses
a_bin = 1.0   # binary semimajor axis in AU (rebound units)

# Simulation constants
P_bin = 2 * np.pi # Binary orbital period for a_bin=1, M_total=1, G=1 (rebound units)
N_periods_short = 750 # Integration time for short simulation, in binary periods
T_short = N_periods_short * P_bin # Total integration time in rebound units
N_output_points = 200 # Number of points to sample during integration for features

# Simulation constants for long integration
N_periods_long = 1e4 # Integration time for long simulation, in binary periods (10^4)
T_long = N_periods_long * P_bin # Total integration time in rebound units

# Define thresholds for instability detection
MIN_DIST_FACTOR = 0.5 # Planet cannot get closer than 0.5*a_bin to primary (or secondary)
ESCAPE_DIST_FACTOR = 2.0 # Planet cannot go further than 2.0*a_bin from primary

def calculate_rho_crit_HW99(mu, e_bin):
    """
    Calculates the critical semimajor axis ratio (rho_crit = a_crit / a_bin)
    for an S-type planet based on the empirical fit from Holman & Wiegert (1999),
    Table 1, for S-type planets.

    Args:
        mu (float): Binary mass ratio (m_B / (m_A + m_B)).
        e_bin (float): Binary eccentricity.

    Returns:
        float: The critical semimajor axis ratio (a_crit / a_bin).
    """
    rho_crit = 0.464 - 0.380 * mu - 0.631 * e_bin + 0.150 * mu * e_bin + 0.198 * e_bin**2 + 0.088 * mu * e_bin**2
    return rho_crit

def run_short_simulation_and_extract_features(params):
    """
    Sets up and runs a short rebound simulation for a given set of parameters,
    then extracts the specified features.

    Args:
        params (dict): A dictionary containing the parameters for one system:
                       'mu', 'e_bin', 'rho', 'e_p', 'inc_p', 'mean_anomaly', 'id'.

    Returns:
        tuple: (dict: extracted features, str: reason for instability or 'ok'). Returns NaNs for features
              if an escape or error occurs.
    """
    mu = params['mu']
    e_bin = params['e_bin']
    rho = params['rho']
    e_p = params['e_p']
    inc_p = params['inc_p'] # Already in radians
    M_p = params.get('M_p', 0.) # Planet mass (default to 0 for test particle)
    mean_anomaly_p = params['mean_anomaly'] # Already in radians
    system_id = params.get('id', 'N/A')

    # Calculate stellar masses
    m_B = mu * M_total
    m_A = M_total - m_B

    # Planetary semimajor axis
    a_p = rho * a_bin

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = 1.0

    sim.add(m=m_A)
    sim.add(m=m_B, primary=sim.particles[0], a=a_bin, e=e_bin, inc=0, Omega=0, omega=0, M=0)
    sim.add(m=M_p, primary=sim.particles[0], a=a_p, e=e_p, inc=inc_p, Omega=0, omega=0, M=mean_anomaly_p)

    sim.move_to_com()
    sim.dt = sim.particles[1].P/20.
    sim.init_megno() # Initialize MEGNO

    times = np.linspace(0, T_short, N_output_points)
    megnos = []
    a_ps = []
    e_vec_x = []
    e_vec_y = []

    features = {
        'mu': mu,
        'e_bin': e_bin,
        'rho': rho,
        'e_p': e_p,
        'inc_p': np.rad2deg(inc_p),
        'megno_median': np.nan,
        'megno_std': np.nan,
        'e_p_free': np.nan,
        'e_p_forced': np.nan,
        'a_p_std': np.nan,
        'rho_crit_HW99': calculate_rho_crit_HW99(mu, e_bin),
        'stable': 0 # Default to unstable, will be set to 1 later if stable
    }
    stability_reason = 'ok'

    try:
        for i, time in enumerate(times):
            sim.integrate(time)
            megnos.append(sim.megno())
            a_ps.append(sim.particles[2].a)
            e_vec_x.append(sim.particles[2].e * np.cos(sim.particles[2].pomega))
            e_vec_y.append(sim.particles[2].e * np.sin(sim.particles[2].pomega))

    except rebound.Escape as error:
        print(f"System {system_id} had an Escape during short integration.")
        stability_reason = 'escape'
        return features, stability_reason
    except rebound.Encounter as error:
        print(f"System {system_id} had an Encounter during short integration.")
        stability_reason = 'encounter'
        return features, stability_reason
    except Exception as e:
        sys.stderr.write(f"System {system_id} had an unexpected error during short integration: {e}\n")
        stability_reason = 'short_error'
        return features, stability_reason

    megnos = np.array(megnos)
    a_ps = np.array(a_ps)
    e_vec_x = np.array(e_vec_x)
    e_vec_y = np.array(e_vec_y)

    features['megno_median'] = np.nanmedian(megnos[int(0.9 * N_output_points):]) if len(megnos) > 0 else np.nan
    features['megno_std'] = np.nanstd(megnos[int(0.2 * N_output_points):]) if len(megnos) > 0 else np.nan

    if len(e_vec_x) > 0:
        mean_e_vec_x = np.nanmean(e_vec_x)
        mean_e_vec_y = np.nanmean(e_vec_y)
        features['e_p_forced'] = np.sqrt(mean_e_vec_x**2 + mean_e_vec_y**2)
        e_p_free_components_x = e_vec_x - mean_e_vec_x
        e_p_free_components_y = e_vec_y - mean_e_vec_y
        e_p_free_instantaneous = np.sqrt(e_p_free_components_x**2 + e_p_free_components_y**2)
        features['e_p_free'] = np.nanmean(e_p_free_instantaneous)
    else:
        features['e_p_forced'] = np.nan
        features['e_p_free'] = np.nan

    initial_a_p = a_p # This is the initial a_p from the input parameters
    features['a_p_std'] = np.nanstd(a_ps) / initial_a_p if initial_a_p != 0 and len(a_ps) > 0 else np.nan

    features['rho_crit_HW99'] = calculate_rho_crit_HW99(mu, e_bin)
    features['stable'] = -1 # Placeholder, will be updated by long integration if short integration was successful

    return features, stability_reason

def run_long_simulation_and_label_stability(initial_params):
    """
    Sets up and runs a long rebound simulation for a given set of parameters
    and determines the stability label. Catches rebound.Escape and rebound.Encounter.

    Args:
        initial_params (dict): A dictionary containing the initial parameters for the system.
                                Expected to contain 'mu', 'e_bin', 'rho', 'e_p', 'inc_p' (radians),
                                'mean_anomaly', 'id'.

    Returns:
        tuple: (int: 1 for stable, 0 for unstable, str: reason for instability or 'stable').
    """
    mu = initial_params['mu']
    e_bin = initial_params['e_bin']
    rho = initial_params['rho']
    e_p = initial_params['e_p']
    inc_p = initial_params['inc_p'] # Already in radians
    M_p = initial_params.get('M_p', 0.)
    mean_anomaly_p = initial_params['mean_anomaly'] # Already in radians
    system_id = initial_params.get('id', 'N/A')

    m_B = mu * M_total
    m_A = M_total - m_B
    a_p = rho * a_bin

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = 1.0

    sim.add(m=m_A)
    sim.add(m=m_B, primary=sim.particles[0], a=a_bin, e=e_bin, inc=0, Omega=0, omega=0, M=0)
    sim.add(m=M_p, primary=sim.particles[0], a=a_p, e=e_p, inc=inc_p, Omega=0, omega=0, M=mean_anomaly_p)

    sim.move_to_com()
    sim.dt = sim.particles[1].P/20.

    # Set up exit conditions for instability
    sim.exit_min_distance = MIN_DIST_FACTOR * a_bin # Minimum distance for planet to any other body (barycenter distance)
    # The rebound.Escape exception handles particles going too far.

    stability_label = 0
    stability_reason = 'long_error' # Default to error until proven otherwise

    try:
        sim.integrate(T_long)
        stability_label = 1 # If integration completes, system is stable
        stability_reason = 'stable'
    except rebound.Escape as error:
        print(f"System {system_id} had an Escape during long integration.")
        stability_reason = 'escape'
    except rebound.Encounter as error:
        print(f"System {system_id} had an Encounter during long integration.")
        stability_reason = 'encounter'
    except Exception as e:
        sys.stderr.write(f"System {system_id} had an unexpected error during long integration: {e}\n")
        stability_reason = 'long_error'

    return stability_label, stability_reason

def worker_simulate_system(system_params):
    """
    Worker function to run both short and long simulations for a single system.
    """
    # First, run short simulation and extract features
    features, short_reason = run_short_simulation_and_extract_features(system_params)

    # If the short simulation already indicated instability (e.g., due to escape/encounter/error),
    # don't run long integration and mark as unstable with the reason from the short sim.
    if short_reason != 'ok':
        features['stable'] = 0
        features['reason'] = short_reason
    else:
        # Then, run long simulation to determine stability label
        stability_label, long_reason = run_long_simulation_and_label_stability(system_params)
        features['stable'] = stability_label
        features['reason'] = long_reason

    return features

def main():
    start_time = time.time() # Record start time
    N_SYSTEMS = int(os.getenv('N_SYSTEMS', '100')) # Default to 100 for quick testing
    OUTPUT_FILENAME = f"s_type_stability_data_{N_SYSTEMS}.csv" # Output file name
    CHECKPOINT_INTERVAL = 500
    N_WORKERS = int(os.getenv('N_WORKERS', '11')) # Default to 11 workers

    print(f"Generating {N_SYSTEMS} systems using {N_WORKERS} workers.")
    print(f"Output will be saved to {OUTPUT_FILENAME} with checkpointing every {CHECKPOINT_INTERVAL} systems.")

    # Generate all initial conditions once
    sampled_params = {
        'mu': np.random.uniform(*mu_range, N_SYSTEMS),
        'e_bin': np.random.uniform(*e_bin_range, N_SYSTEMS),
        'rho': np.random.uniform(*rho_range, N_SYSTEMS),
        'e_p': np.random.uniform(*e_p_range, N_SYSTEMS),
        'inc_p': np.random.uniform(*inc_p_range_rad, N_SYSTEMS),
        'mean_anomaly': np.random.uniform(*mean_anomaly_range_rad, N_SYSTEMS)
    }

    list_of_params_for_workers = []
    for i in range(N_SYSTEMS):
        list_of_params_for_workers.append({
            'id': i, # Add system ID
            'mu': sampled_params['mu'][i],
            'e_bin': sampled_params['e_bin'][i],
            'rho': sampled_params['rho'][i],
            'e_p': sampled_params['e_p'][i],
            'inc_p': sampled_params['inc_p'][i],
            'mean_anomaly': sampled_params['mean_anomaly'][i]
        })

    # Prepare CSV file header if it doesn't exist
    # We need to run one dummy simulation to get all feature keys for the header
    dummy_features = worker_simulate_system(list_of_params_for_workers[0])
    # The 'id' key was temporary for error reporting and should not be a feature in the CSV
    if 'id' in dummy_features:
        del dummy_features['id']
    # The 'reason' key is for internal tracking, not to be saved in CSV
    if 'reason' in dummy_features:
        del dummy_features['reason']
    fieldnames = list(dummy_features.keys())

    file_exists = os.path.isfile(OUTPUT_FILENAME)
    if not file_exists:
        with open(OUTPUT_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Counters for final metrics
    escape_count = 0
    encounter_count = 0
    stable_count = 0
    other_unstable_count = 0 # For short_error, long_error

    # Use multiprocessing Pool
    completed_systems = 0
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        # Use imap_unordered to process results as they come in
        for i, result_features in enumerate(tqdm(
            pool.imap_unordered(worker_simulate_system, list_of_params_for_workers),
            total=N_SYSTEMS, desc="Simulating systems")
        ):
            # Update counts based on the 'reason'
            if result_features['stable'] == 1:
                stable_count += 1
            elif result_features['reason'] == 'escape':
                escape_count += 1
            elif result_features['reason'] == 'encounter':
                encounter_count += 1
            else: # All other unstable reasons (short_error, long_error)
                other_unstable_count += 1

            # Ensure the 'id' and 'reason' keys are not written to the CSV
            if 'id' in result_features:
                del result_features['id']
            if 'reason' in result_features:
                del result_features['reason']

            with open(OUTPUT_FILENAME, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result_features)

            completed_systems += 1
            if completed_systems % CHECKPOINT_INTERVAL == 0:
                sys.stdout.write(f"\nCheckpoint: {completed_systems}/{N_SYSTEMS} systems processed and saved.\n")

    end_time = time.time() # Record end time
    elapsed_time = end_time - start_time

    print(f"\nFinished generating {N_SYSTEMS} systems. Data saved to {OUTPUT_FILENAME}.")
    print(f"\nSummary of results:")
    print(f"  {escape_count} systems had an Escape. (The planet leaves the system without a close approach to a star.)")
    print(f"  {encounter_count} systems had an Encounter. (Close approach to either the primary or secondary star.)")
    print(f"  {stable_count} systems were stable over {N_periods_long} orbits.")
    print(f"  {other_unstable_count} systems were unstable due to other errors (short/long integration). ")
    print(f"Total elapsed time: {elapsed_time / 60:.2f} minutes.")

    # Optional: Display final statistics (only if run in Colab or if script wants to print summary)
    if 'ipykernel' in sys.modules:
        df_final = pd.read_csv(OUTPUT_FILENAME)
        print("\nFinal DataFrame head:")
        display(df_final.head())
        print(f"Total stable systems: {df_final['stable'].sum()} out of {len(df_final)}")
        print(f"Total unstable systems: {len(df_final) - df_final['stable'].sum()} out of {len(df_final)}")


if __name__ == '__main__':
    main()