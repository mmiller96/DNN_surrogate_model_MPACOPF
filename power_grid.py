import pandapower as pp
import numpy as np
from multiprocessing import Pool
from utils import get_DER_sizes
import jax.numpy as jnp

def initialize_3bus_power_grid():
    net = pp.create_empty_network()
    # fixed parameters
    bus1 = pp.create_bus(net, vn_kv=20., name="Bus 1")
    bus2 = pp.create_bus(net, vn_kv=20., name="Bus 2")
    bus3 = pp.create_bus(net, vn_kv=20., name="Bus 3")

    line1 = pp.create_line(net, bus1, bus2, length_km=10, std_type="NAYY 4x50 SE", name="Line 1")
    line2 = pp.create_line(net, bus2, bus3, length_km=10, std_type="NAYY 4x50 SE", name="Line 2")
    line3 = pp.create_line(net, bus1, bus3, length_km=10, std_type="NAYY 4x50 SE", name="Line 3")
    
    # variable parameters
    pp.create_ext_grid(net, bus1, vm_pu=1.00, va_degree=0, name="Grid Connection")
    buses = [bus1, bus2, bus3]
    for i, bus in enumerate(buses):
        pp.create_sgen(net, bus, p_mw=0, q_mvar=0, name=f"Gen {i}")
        pp.create_load(net, bus, p_mw=0, q_mvar=0, name=f"Load {i}")
    net.line['c_nf_per_km'] = 0         # Charging on lines are currently not considered

    config = {  'S_PV_rated':  [2.0, 1.0, 0], 
                'S_WT_rated':  [0, 3.0, 0], 
                'S_BESS_rated':[2.0, 1.0, 0], 
                'S_FC_rated':  [0, 2.0, 0],
                'P_EL_rated':  [0, 2.0, 0],
                'V_slack_limits': [0.95, 1.05],
                'num_nodes': len(net.bus), 
                'num_branches': len(net.line)}
    return net, config

def initialize_ieee_power_grid():
    net = pp.networks.case33bw()
    for bus in net.bus['name'].values:
        pp.create_sgen(net, bus, p_mw=0, q_mvar=0, name=f"Gen {bus}")
    net.load['p_mw'] = 0.0
    net.load['q_mvar'] = 0.0
    net.ext_grid['min_p_mw'] = -100.0
    net.ext_grid['max_p_mw'] = 100.0
    net.ext_grid['min_q_mvar'] = -100.0
    net.ext_grid['max_q_mvar'] = 100.0
    net.bus['max_vm_pu'] = 5.0
    net.bus['min_vm_pu'] = 0.01
    net.line = net.line[net.line['in_service']==True]
    idx_PV = np.array([5, 8, 12, 16, 20, 27, 30])-1
    idx_WT = np.array([14, 22, 24, 27])-1
    idx_RES = np.unique(np.concatenate((idx_PV, idx_WT)))
    idx_BESS = np.array([10, 21, 29])-1
    idx_FC = np.array([10, 21, 29])-1
    idx_EL = np.array([1])-1
    PV_size = np.array([0.7, 0.7, 1.0, 0.8, 1.0, 0.8, 1.0])
    WT_size = np.array([1.5, 1.5, 1.5, 1.5])
    BESS_size = np.array([5.0, 5.0, 5.0])
    BESS_rated = np.array([1.0, 1.0, 1.0])
    FC_size = np.array([1.1, 1.1, 1.1])
    EL_size = np.array([4.0])
    S_PV_rated = dict(zip(idx_PV, PV_size))
    S_WT_rated = dict(zip(idx_WT, WT_size))
    S_BESS_rated = dict(zip(idx_BESS, BESS_rated))
    S_BESS_capacity = dict(zip(idx_BESS, BESS_size))
    S_FC_rated = dict(zip(idx_FC, FC_size))
    P_EL_rated = dict(zip(idx_EL, EL_size)) 
    G, B, lines, B_sh = calculate_line_impedance(net)
    config = {  'S_PV_rated':  get_DER_sizes(S_PV_rated, net.bus.name), 
                'S_WT_rated':  get_DER_sizes(S_WT_rated, net.bus.name), 
                'S_BESS_rated':get_DER_sizes(S_BESS_rated, net.bus.name), 
                'S_BESS_capacity':get_DER_sizes(S_BESS_capacity, net.bus.name), 
                'S_FC_rated':  get_DER_sizes(S_FC_rated, net.bus.name),
                'P_EL_rated':  get_DER_sizes(P_EL_rated, net.bus.name),
                'idx_PV': idx_PV,
                'idx_WT': idx_WT,
                'idx_RES': idx_RES,
                'idx_BESS': idx_BESS,
                'idx_FC': idx_FC,
                'idx_EL': idx_EL,
                'hydrogen_tank': 1000,  # kg
                'V_slack_limits': [0.95, 1.05],
                'cos_phi': 0.95,
                'num_nodes': len(net.bus), 
                'num_branches': len(net.line),
                'eta_ch': 0.95, 
                'eta_dch': 0.98, 
                'alpha_EL': (0.76/40.27)*1000 ,   # MWh/kg 
                'beta_FC': 0.050*1000,
                'G': G,
                'B': B,
                'lines': lines,
                'B_sh': B_sh}
    return net, config

def power_flow_intern(x_indp, net):
    num_nodes = len(net.bus)
    net.sgen['p_mw'].iloc[1:] = x_indp[:num_nodes-1]
    net.sgen['q_mvar'].iloc[1:] = x_indp[num_nodes-1:(num_nodes-1)*2]
    net.ext_grid['vm_pu'] = x_indp[-2]
    net.ext_grid['va_degree'] = x_indp[-1]
    pp.runpp(net, max_iteration=200)

    V = net.res_bus['vm_pu'].values
    angles = net.res_bus['va_degree'].values
    PQ_bus_slack = -net.res_bus[['p_mw', 'q_mvar']].iloc[0].values      # - to correct the direction, + power generation - power consumption
    x_dp = np.hstack((V[1:], angles[1:], PQ_bus_slack))
    return x_dp

def true_x_dp(x_indp, net):
    if x_indp.ndim == 1:
        x_dp = power_flow_intern(x_indp, net)
    elif x_indp.ndim == 2:
        with Pool() as pool:
            x_dp = np.array(pool.starmap(power_flow_intern, [(sample, net) for sample in x_indp]))
    else:
        raise ValueError("Array is not 1 or 2-dimensional")
    return x_dp

def calculate_line_impedance(net):
    """
    Calculate the line impedance parameters for the power grid network.

    Parameters:
    net (object): Power grid network object containing line and bus information.

    Returns:
    G (array): Real part of the admittance for each line.
    B (array): Imaginary part of the admittance for each line.
    lines (array): Array of 'from_bus' and 'to_bus' connections.
    B_sh (array): Shunt susceptance for each line.
    """
    # Calculate line impedances
    net.line = net.line[net.line['in_service']==True]
    Z = net.line['r_ohm_per_km'] * net.line['length_km'] + 1j * net.line['x_ohm_per_km'] * net.line['length_km']
    indices_V = net.line['from_bus'].values
    Z = Z.values / (net.bus['vn_kv'].values[indices_V] ** 2)

    # Calculate admittance (Y) and extract real and imaginary parts
    Y = 1 / Z
    G = np.real(Y)
    B = np.imag(Y)

    # Extract line connections
    lines = net.line[['from_bus', 'to_bus']].values

    # Calculate shunt susceptance for each line
    B_sh = (2 * np.pi * 50 * net.line['length_km'][0] * net.line['c_nf_per_km'][0] * 10 ** -9) * net.bus['vn_kv'].values[indices_V] ** 2
    return G, B, lines, B_sh

def calc_power_flows(V, angle, lines, B, G, B_sh):
    V_i = V[:, lines[:,0]]  # Shape: (num_samples, num_branches)
    V_j = V[:, lines[:,1]]

    # Extract voltage angles for from_bus and to_bus
    angle_i = angle[:, lines[:,0]]
    angle_j = angle[:, lines[:,1]]

    # Compute angle differences in radians
    angle_ij = angle_i - angle_j  # Shape: (num_samples, num_branches)
    angle_ij_rad = jnp.deg2rad(angle_ij)

    G_ij = G.reshape(1, -1)  # Shape: (1, num_branches)
    B_ij = B.reshape(1, -1)
    B_sh = B_sh.reshape(1, -1)

    P_lines = V_i**2 * G_ij - V_i * V_j * (G_ij * jnp.cos(angle_ij_rad) + B_ij * jnp.sin(angle_ij_rad))
    Q_lines = -V_i**2 * (B_ij + B_sh / 2) - V_i * V_j * (G_ij * jnp.sin(angle_ij_rad) - B_ij * jnp.cos(angle_ij_rad))
    PQ_lines = jnp.concat((P_lines, Q_lines), axis=1)
    return PQ_lines

def decouple_V_angles(x_indp, x_dp, num_nodes):
        V_slack = x_indp[:,-2].reshape(-1,1)
        V_tilde = x_dp[:, :num_nodes-1]
        V = jnp.concatenate((V_slack, V_tilde), axis=1)
        angle_tilde = x_dp[:, num_nodes-1:2*(num_nodes-1)]
        angle_slack = x_indp[:,-1].reshape(-1,1)
        angles = jnp.concatenate((angle_slack, angle_tilde), axis=1)
        return V, angles
