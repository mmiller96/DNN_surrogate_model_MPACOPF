import h5py
import pandas as pd
import pickle
import numpy as np
import flax

def save_data(path, data, feature_names, name, delete=True):
    name_columns = name + '_columns'
    save_pickle(path, data, name, delete=delete)
    save_pickle(path, feature_names, name_columns, delete=delete)

def save_pickle(path, x, name, delete=True):
    with h5py.File(path, 'a') as f:
        if delete:
            if name in f:
                del f[name]
        pickled_x= pickle.dumps(x)
        f.create_dataset(name, data=np.void(pickled_x))

def load_pickle(path, name):
    with h5py.File(path, 'r') as f:
        pickled_x = f[name][()]
        x = pickle.loads(pickled_x.tobytes())
        return x
    
def save_NN_model(params, model_NN_pf, path_model):
    attributes = {attr: getattr(model_NN_pf, attr) for attr in model_NN_pf.__annotations__ if attr not in ['parent', 'name']}
    bytes_data = flax.serialization.to_bytes(params)
    with h5py.File(path_model, 'w') as f:
        f.create_dataset('parameters', data=np.void(bytes_data))
        for key, value in attributes.items():
            f.attrs[key] = value
    print(f"Serialized data length: {len(bytes_data)}")

def load_NN_model(path_model):
    with h5py.File(path_model, 'r') as f:
        bytes_data = f['parameters'][()].tobytes() 
        params = flax.serialization.from_bytes(None, bytes_data)
        attr = {key: f.attrs[key] for key in f.attrs}
    return params, attr

def load_carreno_loads(path, num_loads):
    data_carreno = pd.read_csv(path)
    load_columns_Tr = ['Tr. ' + str(i+1) for i in range(num_loads)]
    load_data = data_carreno[load_columns_Tr].copy()
    load_data = load_data*(2/load_data.max().max())
    return load_data, data_carreno

def load_power_flow_dataset(path):
    X_indp_df = load_pickle(path, 'X_indp')
    X_dp_df = load_pickle(path, 'X_dp')
    scaler_X_indp = load_pickle(path, 'scaler_X_indp')
    scaler_X_dp = load_pickle(path, 'scaler_X_dp')
    config = load_pickle(path, 'config')
    X_fix = load_pickle(path, 'X_fix')
    data_df = load_pickle(path, 'data_df')
    net = load_pickle(path, 'net')
    return X_indp_df, X_dp_df, scaler_X_indp, scaler_X_dp, config, X_fix, data_df, net

def print_hdf5_content(path):
    def print_h5_contents(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    with h5py.File(path, 'r') as h5file:
        h5file.visititems(print_h5_contents)