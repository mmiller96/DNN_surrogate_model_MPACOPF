from file_handler import print_hdf5_content, save_pickle, save_pickle
from utils import coeff_PV, coeff_WT
from power_grid import true_x_dp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time


class PowerFlowDatasetGenerator:
    def __init__(self, config_PF, num_samples, batch_size):
        self.S_PV_rated = np.array(config_PF['S_PV_rated'])
        self.S_WT_rated = np.array(config_PF['S_WT_rated'])
        self.S_BESS_rated = np.array(config_PF['S_BESS_rated'])
        self.S_FC_rated = np.array(config_PF['S_FC_rated'])
        self.P_EL_rated = np.array(config_PF['P_EL_rated'])
        self.idx_RES = config_PF['idx_RES']
        self.V_slack_limits = config_PF['V_slack_limits']
        self.cos_phi = config_PF['cos_phi']
        self.num_nodes = config_PF['num_nodes']
        self.num_samples = num_samples
        self.batch_size = batch_size

        if len(self.S_PV_rated) != self.num_nodes:
            raise ValueError(f"Length of S_PV_rated ({len(self.S_PV_rated)}) does not match num_nodes ({self.num_nodes})")
        if len(self.S_WT_rated) != self.num_nodes:
            raise ValueError(f"Length of S_WT_rated ({len(self.S_WT_rated)}) does not match num_nodes ({self.num_nodes})")
        if len(self.S_BESS_rated) != self.num_nodes:
            raise ValueError(f"Length of S_BESS_rated ({len(self.S_BESS_rated)}) does not match num_nodes ({self.num_nodes})")
        if len(self.S_FC_rated) != self.num_nodes:
            raise ValueError(f"Length of S_FC_rated ({len(self.S_FC_rated)}) does not match num_nodes ({self.num_nodes})")
        if len(self.P_EL_rated) != self.num_nodes:
            raise ValueError(f"Length of P_EL_rated ({len(self.P_EL_rated)}) does not match num_nodes ({self.num_nodes})")

    def generate_data_fix(self, path):
        data_df = pd.read_csv(path, index_col=0)
        T = data_df['temperature_2m (°C)'].values
        G =  data_df['shortwave_radiation (W/m²)'].values
        v = data_df['wind_speed_100m (km/h)'].values/ 3.6   # Transform to m/s
        eta_pv, eta_T, eta_G = coeff_PV(T, G)
        eta_WT = coeff_WT(v)
        data_df['eta_PV'] = eta_pv
        data_df['eta_T'] = eta_T
        data_df['eta_G'] = eta_G
        data_df['eta_WT'] = eta_WT
        return data_df
    
    def generate_X_fix(self, data_df):
        S_PV_avail = data_df['eta_PV'].values.reshape(-1,1)*self.S_PV_rated.reshape(1,-1)
        S_WT_avail = data_df['eta_WT'].values.reshape(-1,1)*self.S_WT_rated.reshape(1,-1)
        S_RES_avail = S_PV_avail + S_WT_avail
        S_RES_columns = [f'S_RES_avail_{i}' for i in self.idx_RES]
        S_RES_avail_df = pd.DataFrame(S_RES_avail[:,self.idx_RES], columns=S_RES_columns)
        data_df = data_df.reset_index(drop=True)
        X_fix_df = pd.concat((data_df.iloc[:,:2*self.num_nodes], S_RES_avail_df), axis=1)
        return X_fix_df, data_df

    def generate_bus_boundaries(self, X_fix_df):
        PQ_L_time_series_df = X_fix_df.iloc[:,:2*self.num_nodes]
        P_bus_lower_tilde = (-PQ_L_time_series_df.iloc[:,1:33]).min(axis=0).values - self.S_BESS_rated[1:33]
        P_bus_upper_tilde = (-PQ_L_time_series_df.iloc[:,1:33]).max(axis=0).values + self.S_PV_rated[1:33]+self.S_WT_rated[1:33] + self.S_FC_rated[1:33] + self.S_BESS_rated[1:33]
        Q_bus_lower_tilde = (-PQ_L_time_series_df.iloc[:,34:]).min(axis=0).values - (self.S_PV_rated[1:33]+self.S_WT_rated[1:33] + self.S_FC_rated[1:33] + self.S_BESS_rated[1:33])*np.tan(np.arccos(self.cos_phi))
        Q_bus_upper_tilde = (-PQ_L_time_series_df.iloc[:,34:]).max(axis=0).values + (self.S_PV_rated[1:33]+self.S_WT_rated[1:33] + self.S_FC_rated[1:33] + self.S_BESS_rated[1:33])*np.tan(np.arccos(self.cos_phi))
        return P_bus_lower_tilde, P_bus_upper_tilde, Q_bus_lower_tilde, Q_bus_upper_tilde
    
    def generate_dataset(self, net, X_fix_df):
        P_bus_lower_tilde, P_bus_upper_tilde, Q_bus_lower_tilde, Q_bus_upper_tilde = self.generate_bus_boundaries(X_fix_df)
        P_bus_tilde = np.random.uniform(low=P_bus_lower_tilde, 
                                high=P_bus_upper_tilde, 
                                size=(self.num_samples, self.num_nodes-1))

        Q_bus_tilde = np.random.uniform(low=Q_bus_lower_tilde, 
                                        high=Q_bus_upper_tilde, 
                                        size=(self.num_samples, self.num_nodes-1))
        V_slack = np.random.uniform(self.V_slack_limits[0], self.V_slack_limits[1], self.num_samples)
        angle_slack = np.zeros(self.num_samples)
        V_angle_slack = pd.DataFrame({
            'V_slack': V_slack,
            'angle_slack': angle_slack
        })
        start_time = time.time()
        X_indp, X_dp = self.simulate_power_flow(P_bus_tilde, Q_bus_tilde, V_angle_slack, net)
        end_time = time.time() 
        execution_time = end_time - start_time  
        print(f"Power flow simulation time: {execution_time:.4f} seconds")
        return X_indp, X_dp
    
    def simulate_power_flow(self, P_bus_tilde, Q_bus_tilde, V_angle_slack, net): 
        X_dp = []
        num_batches = self.num_samples // self.batch_size
        X_indp_values = np.hstack((P_bus_tilde, Q_bus_tilde, V_angle_slack))
        for batch in range(num_batches):
            print(f"Processing batch {batch + 1}/{num_batches}...")
            start_idx = batch * self.batch_size
            end_idx = start_idx + self.batch_size
            X_indp_batch = X_indp_values[start_idx:end_idx]
            X_dp_batch = true_x_dp(X_indp_batch, net)
            X_dp.append(X_dp_batch)  

        # Handle the remaining samples if num_samples is not a multiple of batch_size
        remaining_samples = self.num_samples % self.batch_size
        if remaining_samples > 0:
            print(f"Processing remaining {remaining_samples} samples...")
            X_indp_batch = X_indp_values[-remaining_samples:end_idx]
            X_dp_batch = true_x_dp(X_indp_batch, net)
            X_dp.append(X_dp_batch)
        X_indp_columns = [f'P_bus{i}' for i in range(1, self.num_nodes)] + [f'Q_bus{i}' for i in range(1, self.num_nodes)] + ['V_slack', 'angle_slack']
        X_indp = pd.DataFrame(X_indp_values, columns=X_indp_columns)
        X_dp = pd.DataFrame(np.vstack(X_dp), columns=self.get_X_dp_columns())
        return X_indp, X_dp
    
    def create_scalers(self, X_indp, X_dp):
        scaler_X_indp = StandardScaler().fit(X_indp)
        scaler_X_dp = StandardScaler().fit(X_dp)
        scalers = {'X_indp': scaler_X_indp, 'X_dp': scaler_X_dp}
        return scalers
    
    def get_X_dp_columns(self):
        X_dp_cols = []
        for i in range(1, self.num_nodes):
            X_dp_cols.extend([f"V_{i}"])
        for i in range(1, self.num_nodes):
            X_dp_cols.extend([f"angle_{i}"]) 
        X_dp_cols.extend([f"P_bus_slack", "Q_bus_slack"])
        return X_dp_cols
    
    def save_dataset(self, X_indp, X_dp, scalers, net, data_df, X_fix_df, config, path):
        save_pickle(path, X_indp, 'X_indp')
        save_pickle(path, X_dp, 'X_dp')
        save_pickle(path, config, 'config')
        save_pickle(path, scalers['X_indp'], 'scaler_X_indp')
        save_pickle(path, scalers['X_dp'], 'scaler_X_dp')
        save_pickle(path, net, 'net')
        save_pickle(path, X_fix_df, 'X_fix')
        save_pickle(path, data_df, 'data_df')
        print_hdf5_content(path)