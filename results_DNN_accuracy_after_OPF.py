from file_handler import load_pickle, load_power_flow_dataset
from power_grid import true_x_dp, decouple_V_angles, calculate_line_impedance, calc_power_flows
from results_DNN_accuracy import load_data_and_model
import numpy as np
import os

if __name__ == '__main__':
    name = 'ieee33'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(script_dir, "data", "ieee33dataset.h5")
    path_model = os.path.join(script_dir, "models", "ieee33_DNN_best.h5")
    path_result = os.path.join(script_dir, "results", "ieee33results_April.h5")

    X_pf_df, Y_pf_df, scaler_X_pf, scaler_Y_pf, config, X_opf_fix, data_df, net = load_power_flow_dataset(path_dataset)
    config = {
        'path': path_dataset,
        'lr': 2e-5,
        'test_size': 0.15,
        'batch_size': 1,
        'random_state': 42
    }
    X_indp_test, X_dp_test, params, model_NN_pf, net, num_nodes, functions = load_data_and_model(config, path_model)
    results = load_pickle(path_result, name)
    V = results['V']
    angle = results['angle']
    P_line = results['P_line']
    Q_line = results['Q_line']
    P_ext = results['P_ext']
    Q_ext = results['Q_ext']
    x_dp_true = true_x_dp(results['x_indp'], net)
    V_true, angles_true = decouple_V_angles(results['x_indp'], x_dp_true, num_nodes=len(net.bus))
    V_diff = 1e3*np.abs(V_true[:,1:]-V[:,1:])
    angle_diff = 1e3*np.abs(angles_true[:,1:]-angle[:,1:])
    print('Voltage diff: ' + str(V_diff.mean(axis=0)))
    print('Angle diff: ' + str(angle_diff.mean(axis=0)))
    print('Voltage mean: ' + str(V_diff.mean()))
    print('Angle mean: ' + str(angle_diff.mean()))
    
    G, B, lines, B_sh = calculate_line_impedance(net)
    PQ_lines_true = calc_power_flows(V_true, angles_true, lines, B, G, B_sh)
    P_lines_true = PQ_lines_true[:,:32]
    Q_lines_true = PQ_lines_true[:,32:]
    S_lines_true = np.sqrt(P_lines_true**2 + Q_lines_true**2)
    P_err = np.abs(P_line - P_lines_true)/np.abs(S_lines_true+1e-9)
    Q_err = np.abs(Q_line - Q_lines_true)/np.abs(S_lines_true+1e-9)
    print('P_line error: ' + str(P_err.mean(axis=0)))
    print('Q_line error: ' + str(Q_err.mean(axis=0)))
    print('P_line error mean: ' + str(P_err.mean()))
    print('Q_line error mean: ' + str(Q_err.mean()))