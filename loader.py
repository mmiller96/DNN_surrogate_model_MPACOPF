from file_handler import load_power_flow_dataset
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from jax import device_put
import numpy as np

class Loader_PF:
    def __init__(self, config):
        self.path = config['path']
        self.test_size = config['test_size']
        self.batch_size = config['batch_size']

    def initialize_data(self):
        X_indp_df, X_dp_df, scaler_X_indp, scaler_X_dp, _, _, _, _ = load_power_flow_dataset(self.path)
        X_indp_train, X_indp_val_test, X_dp_train, X_dp_val_test = train_test_split(X_indp_df.values, X_dp_df.values, test_size=2*self.test_size)
        X_indp_val, X_indp_test, X_dp_val, X_dp_test = train_test_split(X_indp_val_test, X_dp_val_test, test_size=0.5)
        
        X_indp_train = device_put(jnp.array(X_indp_train))
        X_dp_train = device_put(jnp.array(X_dp_train))
        X_indp_val = device_put(jnp.array(X_indp_val))
        X_dp_val = device_put(jnp.array(X_dp_val))
        X_indp_test = device_put(jnp.array(X_indp_test))
        X_dp_test = device_put(jnp.array(X_dp_test))
        data = {
            'X_indp_train': X_indp_train,
            'X_dp_train': X_dp_train,
            'X_indp_val': X_indp_val,
            'X_dp_val': X_dp_val,
            'X_indp_test': X_indp_test,
            'X_dp_test': X_dp_test
        }
        scalers = {
            'scaler_X_indp': scaler_X_indp,
            'scaler_X_dp': scaler_X_dp
        }
        return data, scalers

    def create_data_loader(self, X_data, y_data, batch_size):
        num_samples = X_data.shape[0]
        indices = np.arange(num_samples)
        
        def data_loader():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                yield X_data[batch_indices], y_data[batch_indices]
        
        return data_loader()