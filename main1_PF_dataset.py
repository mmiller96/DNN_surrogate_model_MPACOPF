from models1_PF_dataset import PowerFlowDatasetGenerator
from power_grid import initialize_ieee_power_grid
import os

if __name__== '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(script_dir, "data", "ieee33.csv")
    path_dataset = os.path.join(script_dir, "data", "ieee33dataset.h5")

    net, config = initialize_ieee_power_grid()
    model_PF = PowerFlowDatasetGenerator(config, num_samples=500000, batch_size=100000)
    data_fix_df = model_PF.generate_data_fix(path=path_data)
    X_fix_df, data_fix_df = model_PF.generate_X_fix(data_fix_df)
    X_indp, X_dp = model_PF.generate_dataset(net, X_fix_df)
    scalers = model_PF.create_scalers(X_indp, X_dp)
    model_PF.save_dataset(X_indp, X_dp, scalers, net, data_fix_df, X_fix_df, config, path=path_dataset)