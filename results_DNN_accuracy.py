from models2_PF_training import PowerFlowModel
from power_grid import (
    power_flow_intern,
    initialize_ieee_power_grid,
    calculate_line_impedance,
    calc_power_flows
)
from models_NN import NN_pf
from file_handler import load_NN_model
from functions import define_pf_functions
from loader import Loader_PF
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import jax
import pdb
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    # Optionally specify the LaTeX preamble to include specific packages
    "text.latex.preamble": r"\usepackage{amsmath}",
})
size = 40
jax.config.update('jax_platform_name', 'cpu')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(script_dir, "data", "ieee33dataset.h5")
    path_model = os.path.join(script_dir, "models", "ieee33_DNN_best.h5")
    #name = 'ieee33'
    config = {
        'path': path_dataset,
        'lr': 2e-5,
        'test_size': 0.15,
        'batch_size': 1,
        'random_state': 42
    }
    num_samples = 1000

    # Load data and model
    X_indp_test, X_dp_test, params, model_NN_pf, net, num_nodes, functions = load_data_and_model(config, path_model)
    
    # Run power flow simulations
    X_dp_test_pred, X_dp_test_true = run_power_flow_simulations(X_indp_test, params, functions, net, num_nodes, num_samples)
    # Calculate line flows
    V_pred, V_true, angle_pred, angle_true = extract_voltages_angles(X_indp_test, X_dp_test_pred, X_dp_test_true, num_nodes, num_samples)
    PQ_line_true, PQ_line_pred = calculate_line_flows(V_pred, V_true, angle_pred, angle_true, net)
    PQ_line_mean = np.abs(PQ_line_true).mean(axis=0)
    P_line_mean = PQ_line_mean[:32]
    Q_line_mean = PQ_line_mean[32:]
    P_line_mean_ratio = P_line_mean/P_line_mean.mean()
    Q_line_mean_ratio = Q_line_mean/Q_line_mean.mean()
    print('Active power: ' + str(100*P_line_mean_ratio[16]) + '%    ' + str(str(100*P_line_mean_ratio[-1])) + '%')  
    print('Reactive power:: ' + str(100*Q_line_mean_ratio[16]) + '%    ' + str(str(100*Q_line_mean_ratio[-1])) + '%')
    # roughly 20 times bigger
    voltage_errors, angle_errors = compute_voltage_angle_errors(V_true, V_pred, angle_true, angle_pred)
    P_errors, Q_errors = compute_line_flow_errors(PQ_line_true[:, :32], PQ_line_pred[:, :32], PQ_line_true[:, 32:], PQ_line_pred[:, 32:])
    plot_voltage_angle_errors(voltage_errors, angle_errors, script_dir)
    plot_line_flow_errors(P_errors, Q_errors, net, script_dir)

def load_data_and_model(config, path_model):
    """Load test data and the trained neural network model."""
    loader = Loader_PF(config)
    data, scalers = loader.initialize_data()
    X_indp_test, X_dp_test = data['X_indp_test'], data['X_dp_test']
    params, attr = load_NN_model(path_model)
    model_NN_pf = NN_pf(**attr)
    net, _ = initialize_ieee_power_grid()
    num_nodes = len(net.bus)
    model = PowerFlowModel(config)
    optimizer, _, _ = model.initialize_PF_optimizer(model_NN_pf)
    functions = define_pf_functions(optimizer, model_NN_pf, model.scalers)

    return X_indp_test, X_dp_test, params, model_NN_pf, net, num_nodes, functions

def run_power_flow_simulations(X_indp_test, params, functions, net, num_nodes, num_samples):
    """Run power flow simulations using the neural network and pandapower."""
    # Run simulations with the neural network
    start_time = time.time()
    x_dp_pred = []
    for i in range(num_samples):
        x_dp_sample = functions['power_flow'](params, X_indp_test[i])
        x_dp_pred.append(x_dp_sample)
    x_dp_pred = np.array(x_dp_pred)
    nn_execution_time = time.time() - start_time
    print(f"Execution time NN: {nn_execution_time:.4f} seconds")

    # Run simulations with pandapower
    X_indp_test = np.array(X_indp_test)
    start_time = time.time()
    x_dp_true = []
    for i in range(num_samples):
        x_dp_true_sample = power_flow_intern(X_indp_test[i], net)
        x_dp_true.append(x_dp_true_sample)
    x_dp_true = np.array(x_dp_true)
    pp_execution_time = time.time() - start_time
    print(f"Execution time pandapower: {pp_execution_time:.4f} seconds")
    return x_dp_pred, x_dp_true

def extract_voltages_angles(X_indp, X_dp_pred, X_dp_true, num_nodes, num_samples):
    """Extract voltages and angles from predictions and true values."""
    # Include slack bus voltages and angles
    V_pred = np.hstack((X_indp[:num_samples, -2].reshape(-1, 1), X_dp_pred[:, :num_nodes - 1]))
    V_true = np.hstack((X_indp[:num_samples, -2].reshape(-1, 1), X_dp_true[:, :num_nodes - 1]))
    angle_pred = np.hstack((X_indp[:num_samples, -1].reshape(-1, 1), X_dp_pred[:, num_nodes - 1:2 * (num_nodes - 1)]))
    angle_true = np.hstack((X_indp[:num_samples, -1].reshape(-1, 1), X_dp_true[:, num_nodes - 1:2 * (num_nodes - 1)]))
    return V_pred, V_true, angle_pred, angle_true

def calculate_line_flows(V_pred, V_true, angle_pred, angle_true, net):
    """Calculate line power flows based on voltages and angles."""
    G, B, lines, B_sh = calculate_line_impedance(net)
    PQ_line_true = calc_power_flows(V_true, angle_true, lines, B, G, B_sh)
    PQ_line_pred = calc_power_flows(V_pred, angle_pred, lines, B, G, B_sh)
    return PQ_line_true, PQ_line_pred

def compute_voltage_angle_errors(V_true, V_pred, angle_true, angle_pred):
    """Compute absolute errors for voltages and angles."""
    voltage_errors = np.abs(V_true - V_pred)*1e3
    angle_errors = np.abs(angle_true - angle_pred)*1e3
    return voltage_errors, angle_errors

def compute_line_flow_errors(P_line_true, P_line_pred, Q_line_true, Q_line_pred):
    """Compute absolute errors for line power flows per sample per line."""
    S_line_true = np.sqrt(P_line_true**2 + Q_line_true**2)
    P_errors = np.abs(P_line_true - P_line_pred) / (np.abs(S_line_true) + 1e-9)
    Q_errors = np.abs(Q_line_true - Q_line_pred) / (np.abs(S_line_true) + 1e-9)
    return P_errors, Q_errors


def plot_line_flow_errors(P_errors, Q_errors, net, script_dir):
    """Plot absolute error distribution for active and reactive power flow predictions per line."""
    line_labels = [f"{row['from_bus']+1} - {row['to_bus']+1}" for _, row in net.line.iterrows()]
    num_lines = len(line_labels)
    
    P_error_list = [P_errors[:, i] for i in range(num_lines)]
    Q_error_list = [Q_errors[:, i] for i in range(num_lines)]
    
    positions = np.arange(num_lines)
    positions_P = positions - 0.2  # Shift active power boxes slightly to the left
    positions_Q = positions + 0.2  # Shift reactive power boxes slightly to the right

    fig, ax1 = plt.subplots(figsize=(20, 8))
    bp_P = ax1.boxplot(
        P_error_list, positions=positions_P, widths=0.35,
        patch_artist=True, labels=line_labels, showfliers=False
    )
    ax2 = ax1.twinx()
    bp_Q = ax2.boxplot(
        Q_error_list, positions=positions_Q, widths=0.35,
        patch_artist=True, labels=line_labels, showfliers=False
    )

    # Customize colors
    customize_boxplot(bp_P, 'red', 'lightcoral')
    customize_boxplot(bp_Q, 'green', 'lightseagreen')

    # Set labels and title
    ax1.set_xticks(positions)
    ax1.set_xticklabels(line_labels, rotation='vertical', fontsize=size*0.8)
    ax1.tick_params(axis='y', labelcolor='red', labelsize=size)
    ax2.tick_params(axis='y', labelcolor='green', labelsize=size)
    ax1.set_ylim(0, 0.23)  # For active power
    ax2.set_ylim(0, 0.23)  
    # Add legend
    legend_elements = [
        Patch(facecolor='lightcoral', edgecolor='red', label=r'$\frac{|\hat{P}_{ij} - P_{ij}|}{|S_{ij}|}$'),
        Patch(facecolor='lightseagreen', edgecolor='green', label=r'$\frac{|\hat{Q}_{ij} - Q_{ij}|}{|S_{ij}|}$')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=size)

    plt.tight_layout()
    plt.grid()
    path_save = os.path.join(script_dir, "results", "pictures", "documentation", "NN_Power_Flow_Error.pdf")
    plt.savefig(path_save)
    plt.close()


def plot_voltage_angle_errors(voltage_errors, angle_errors, script_dir):
    """Plot absolute error distribution for voltage and angle predictions per bus."""
    num_buses = voltage_errors.shape[1]
    labels = [f'{i}' for i in range(1, num_buses + 1)]
    
    # Prepare data for box plots
    voltage_error_list = [voltage_errors[:, i] for i in range(1, num_buses)]
    angle_error_list = [angle_errors[:, i] for i in range(1, num_buses)]

    positions = np.arange(1, len(labels))
    positions_voltage = positions - 0.2
    positions_angle = positions + 0.2

    fig, ax1 = plt.subplots(figsize=(20, 8))
    bp_voltage = ax1.boxplot(
        voltage_error_list, positions=positions_voltage, widths=0.35,
        patch_artist=True, labels=labels[1:], showfliers=False, 
    )
    ax2 = ax1.twinx()
    bp_angle = ax2.boxplot(
        angle_error_list, positions=positions_angle, widths=0.35,
        patch_artist=True, labels=labels[1:], showfliers=False
    )
    customize_boxplot(bp_voltage, 'blue', 'lightblue')
    customize_boxplot(bp_angle, 'green', 'lightgreen')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels[1:], rotation='horizontal', fontsize=size*0.8)
    ax1.set_xlabel('Bus', fontsize=size)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=size)
    ax2.tick_params(axis='y', labelcolor='green', labelsize=size)

    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', label=r'$|\hat{V}_i - V_i|\cdot 10^{3}$ [p.u.]'),
        Patch(facecolor='lightgreen', edgecolor='green', label=r'$|\hat{\delta}_i - \delta_i|\cdot 10^{3}$ [$^\circ$]')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=size)

    plt.tight_layout()
    plt.grid()
    path_save = os.path.join(script_dir, "results", "pictures", "documentation", "NN_Voltage_Angle_Error.pdf")
    plt.savefig(path_save)
    plt.close()


def customize_boxplot(bp, edge_color, fill_color):
    """Customize boxplot appearance."""
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set_facecolor(fill_color)
    for flier in bp['fliers']:
        flier.set(marker='o', color=edge_color, alpha=0.5)


if __name__ == '__main__':
    main()