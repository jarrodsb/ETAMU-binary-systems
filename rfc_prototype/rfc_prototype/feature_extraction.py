# writefile rfc_prototype/rfc_prototype/feature_extraction.py

import rebound
import numpy as np

# Global constants for feature extraction
M_total = 1.0 # total binary mass in solar masses
a_bin = 10.0   # binary semimajor axis in AU (rebound units)

# Simulation constants for short integration
P_bin = 2 * np.pi * np.sqrt(a_bin**3 / M_total) # Binary orbital period for a_bin=1, M_total=1, G=1
N_periods_short = 750 # Integration time for short simulation, in binary periods
T_short = N_periods_short * P_bin # Total integration time in rebound units#
N_output_points = 200 # Number of points to sample during integration for features
M_p_ref_for_features = 1e-5 * M_total # A small, non-zero mass for feature calculations

def calculate_rho_crit_HW99(mu, e_bin):
    """
    Calculates the critical semimajor axis ratio (rho_crit = a_crit / a_bin)
    for an S-type planet based on the empirical fit from Holman & Wiegert (1999),
    Table 1, for S-type planets.
    """
    rho_crit = 0.464 - 0.380 * mu - 0.631 * e_bin + 0.150 * mu * e_bin + 0.198 * e_bin**2 + 0.088 * mu * e_bin**2
    return rho_crit

def compute_features(sim: rebound.Simulation, mu: float, e_bin: float) -> dict:
    """
    Extracts specified features from a rebound.Simulation object for a two-planet system
    after a short integration.

    The sim object is assumed to be already set up with 2 stars and 2 planets.
    sim.particles[0] = primary star
    sim.particles[1] = secondary star
    sim.particles[2] = planet 1
    sim.particles[3] = planet 2

    Args:
        sim (rebound.Simulation): The rebound simulation object, already initialized
                                  and set to the initial conditions of the system.
        mu (float): The binary mass ratio.
        e_bin (float): The binary eccentricity.

    Returns:
        dict: A dictionary containing the extracted features. Returns NaNs for features
              if an escape, encounter, or other error occurs during the short integration.
    """

    # Initialize MEGNO if not already done
    if not sim.megno_initialized:
        sim.init_megno()

    # Get initial planetary parameters (needed for features and normalizations)
    a_p1_initial = sim.particles[2].a
    e_p1_initial = sim.particles[2].e
    inc_p1_initial_rad = sim.particles[2].inc

    a_p2_initial = sim.particles[3].a
    e_p2_initial = sim.particles[3].e
    inc_p2_initial_rad = sim.particles[3].inc

    # Derive rho values from initial semimajor axes
    rho_p1_initial = a_p1_initial / a_bin
    rho_p2_initial = a_p2_initial / a_bin

    times = np.linspace(0, T_short, N_output_points)

    # Planet 1 data
    megnos_p1 = []
    a_ps_p1 = []
    e_vec_x_p1 = []
    e_vec_y_p1 = []
    inc_ps_p1 = []

    # Planet 2 data
    megnos_p2 = []
    a_ps_p2 = []
    e_vec_x_p2 = []
    e_vec_y_p2 = []
    inc_ps_p2 = []

    features = {
        'mu': mu,
        'e_bin': e_bin,
        'rho_crit_HW99_p1': calculate_rho_crit_HW99(mu, e_bin),

        # Planet 1 features
        'rho_p1': rho_p1_initial,
        'e_p1': e_p1_initial,
        'inc_p_deg_p1': np.rad2deg(inc_p1_initial_rad),
        'megno_median_p1': np.nan,
        'megno_std_p1': np.nan,
        'e_p_free_p1': np.nan,
        'e_p_forced_p1': np.nan,
        'a_p_std_p1': np.nan,

        # Planet 2 features
        'rho_p2': rho_p2_initial,
        'e_p2': e_p2_initial,
        'inc_p_deg_p2': np.rad2deg(inc_p2_initial_rad),
        'megno_median_p2': np.nan,
        'megno_std_p2': np.nan,
        'e_p_free_p2': np.nan,
        'e_p_forced_p2': np.nan,
        'a_p_std_p2': np.nan,

        # Planet-planet interaction features
        'period_ratio_p2_p1': np.nan,
        'initial_separation_au': np.nan,
        'init_mutual_inc_deg': np.nan,
        'mmr_flag_2_1': 0,
        'mmr_flag_3_2': 0
    }

    try:
        for i, time in enumerate(times):
            sim.integrate(time)

            system_megno = sim.megno()
            megnos_p1.append(system_megno)
            megnos_p2.append(system_megno)

            # Planet 1 (index 2)
            a_ps_p1.append(sim.particles[2].a)
            e_vec_x_p1.append(sim.particles[2].e * np.cos(sim.particles[2].pomega))
            e_vec_y_p1.append(sim.particles[2].e * np.sin(sim.particles[2].pomega))
            inc_ps_p1.append(sim.particles[2].inc)

            # Planet 2 (index 3)
            a_ps_p2.append(sim.particles[3].a)
            e_vec_x_p2.append(sim.particles[3].e * np.cos(sim.particles[3].pomega))
            e_vec_y_p2.append(sim.particles[3].e * np.sin(sim.particles[3].pomega))
            inc_ps_p2.append(sim.particles[3].inc)

    except (rebound.Escape, rebound.Encounter, Exception) as error:
        # If any instability occurs, return NaN for all features
        return features # features dict already initialized with NaNs for most values

    # Convert lists to numpy arrays for calculations
    megnos_p1 = np.array(megnos_p1)
    a_ps_p1 = np.array(a_ps_p1)
    e_vec_x_p1 = np.array(e_vec_x_p1)
    e_vec_y_p1 = np.array(e_vec_y_p1)
    inc_ps_p1 = np.array(inc_ps_p1)

    megnos_p2 = np.array(megnos_p2)
    a_ps_p2 = np.array(a_ps_p2)
    e_vec_x_p2 = np.array(e_vec_x_p2)
    e_vec_y_p2 = np.array(e_vec_y_p2)
    inc_ps_p2 = np.array(inc_ps_p2)

    # Calculate features for Planet 1
    features['megno_median_p1'] = np.nanmedian(megnos_p1[int(0.9 * N_output_points):]) if len(megnos_p1) > 0 else np.nan
    features['megno_std_p1'] = np.nanstd(megnos_p1[int(0.2 * N_output_points):]) if len(megnos_p1) > 0 else np.nan

    if len(e_vec_x_p1) > 0:
        mean_e_vec_x_p1 = np.nanmean(e_vec_x_p1)
        mean_e_vec_y_p1 = np.nanmean(e_vec_y_p1)
        features['e_p_forced_p1'] = np.sqrt(mean_e_vec_x_p1**2 + mean_e_vec_y_p1**2)
        e_p_free_components_x_p1 = e_vec_x_p1 - mean_e_vec_x_p1
        e_p_free_components_y_p1 = e_vec_y_p1 - mean_e_vec_y_p1
        e_p_free_instantaneous_p1 = np.sqrt(e_p_free_components_x_p1**2 + e_p_free_components_y_p1**2)
        features['e_p_free_p1'] = np.nanmean(e_p_free_instantaneous_p1)
    else:
        features['e_p_forced_p1'] = np.nan
        features['e_p_free_p1'] = np.nan

    features['a_p_std_p1'] = np.nanstd(a_ps_p1) / a_p1_initial if a_p1_initial != 0 and len(a_ps_p1) > 0 else np.nan

    # Calculate features for Planet 2
    features['megno_median_p2'] = np.nanmedian(megnos_p2[int(0.9 * N_output_points):]) if len(megnos_p2) > 0 else np.nan
    features['megno_std_p2'] = np.nanstd(megnos_p2[int(0.2 * N_output_points):]) if len(megnos_p2) > 0 else np.nan

    if len(e_vec_x_p2) > 0:
        mean_e_vec_x_p2 = np.nanmean(e_vec_x_p2)
        mean_e_vec_y_p2 = np.nanmean(e_vec_y_p2)
        features['e_p_forced_p2'] = np.sqrt(mean_e_vec_x_p2**2 + mean_e_vec_y_p2**2)
        e_p_free_components_x_p2 = e_vec_x_p2 - mean_e_vec_x_p2
        e_p_free_components_y_p2 = e_vec_y_p2 - mean_e_vec_y_p2
        e_p_free_instantaneous_p2 = np.sqrt(e_p_free_components_x_p2**2 + e_p_free_components_y_p2**2)
        features['e_p_free_p2'] = np.nanmean(e_p_free_instantaneous_p2)
    else:
        features['e_p_forced_p2'] = np.nan
        features['e_p_free_p2'] = np.nan

    features['a_p_std_p2'] = np.nanstd(a_ps_p2) / a_p2_initial if a_p2_initial != 0 and len(a_ps_p2) > 0 else np.nan

    # Calculate Planet-Planet Interaction Features
    if sim.particles[2].P > 0 and sim.particles[3].P > 0:
        features['period_ratio_p2_p1'] = sim.particles[3].P / sim.particles[2].P
    else:
        features['period_ratio_p2_p1'] = np.nan

    features['initial_separation_au'] = a_p2_initial - a_p1_initial # Assumes a_p2 > a_p1

    # Mutual inclination (assuming initial Omega=0 for both)
    features['init_mutual_inc_deg'] = np.rad2deg(np.abs(inc_ps_p1[-1] - inc_ps_p2[-1])) # Using final inclinations

    # Mean motion resonance flags (initial state)
    if not np.isnan(features['period_ratio_p2_p1']):
        if 1.95 <= features['period_ratio_p2_p1'] <= 2.05:
            features['mmr_flag_2_1'] = 1
        if 1.45 <= features['period_ratio_p2_p1'] <= 1.55:
            features['mmr_flag_3_2'] = 1

    return features
