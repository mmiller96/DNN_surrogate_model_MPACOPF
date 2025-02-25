import numpy as np
import jax.numpy as jnp

def coeff_PV(T, G, NOCT=45, eta_conv_PV=0.95, p_coef=-0.35):
    T_GT = T + (NOCT - 20) * G / 800
    Delta_T = T_GT - 25
    Delta_P = Delta_T * p_coef
    eta_T = 1 + (Delta_P / 100)
    eta_G = G / 1000
    return eta_T * eta_G * eta_conv_PV, eta_T, eta_G

def coeff_WT(v, v_cut_in=3, v_rated=10.3, v_cut_out=22):
    """
    Calculate the wind turbine efficiency coefficient (eta_WT) for each wind speed in an array,
    using coefficients a and b for the power curve.

    Parameters:
    - v: Array of wind speeds (m/s)
    - v_rated: Rated wind speed (m/s)
    - v_cut_in: Cut-in wind speed (m/s)
    - v_cut_out: Cut-out wind speed (m/s)
    - p_rated: Rated power (MW)

    Returns:
    - eta_WT: Array of coefficients to multiply with rated power to get actual power output.
    """
    # Ensure v is a NumPy array
    v = np.array(v)
    a = 1 / (v_rated ** 3 - v_cut_in ** 3)
    b = v_cut_in ** 3 / (v_rated ** 3 - v_cut_in ** 3)
    eta_WT = np.zeros_like(v)

    below_cut_in = v < v_cut_in
    eta_WT[below_cut_in] = 0

    between_cut_in_and_rated = (v_cut_in <= v) & (v < v_rated)
    eta_WT[between_cut_in_and_rated] = a * (v[between_cut_in_and_rated] ** 3) - b

    between_rated_and_cut = (v_rated <= v) & (v <= v_cut_out)
    eta_WT[between_rated_and_cut] = 1
    above_cut = v > v_cut_out
    eta_WT[above_cut] = 0
    return eta_WT

def get_DER_sizes(x_dict, net_bus_names):
    x_flipped = {v: k for k, v in net_bus_names.items()}
    DER_vector = [0.0] * len(net_bus_names)
    for bus_name, value in x_dict.items():
        DER_vector[x_flipped[bus_name]] = value
    return DER_vector

def transform_to_numpy(results_dict):
    """
    Transforms the values in the dictionary to NumPy arrays.

    Parameters:
    results_dict (dict): The dictionary containing the data.

    Returns:
    dict: A new dictionary with all values converted to NumPy arrays.
    """
    numpy_dict = {}
    for key, value in results_dict.items():
        if isinstance(value, jnp.ndarray):
            # Convert JAX array to NumPy array
            numpy_dict[key] = np.array(value)
        elif isinstance(value, (list, np.ndarray)):
            numpy_dict[key] = np.array(value)
        else:
            numpy_dict[key] = value
    return numpy_dict
