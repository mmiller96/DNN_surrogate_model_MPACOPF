from models2_PF_training import PowerFlowModel
from models_NN import NN_pf
from file_handler import load_NN_model, save_NN_model
from functions import define_pf_functions
import jax
from power_grid import initialize_ieee_power_grid
from jax import config
import time
import os

if __name__== '__main__':
    print(jax.default_backend())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(script_dir, "data", "ieee33dataset.h5")
    path_model = os.path.join(script_dir, "models", "ieee33_DNN_best.h5")
    net, config = initialize_ieee_power_grid()

    config = {'path':path_dataset, 'test_size':0.15, 'batch_size':1000, 'random_state':42, 'lr':2e-5, 'num_epochs': 10000}
    model = PowerFlowModel(config)
    load_model = False
    if load_model:
        _, attr = load_NN_model(path_model)
    else:
        attr = {'input_size':model.input_size, 'hidden_sizes':[200, 300, 200], 'output_size': model.output_size}
    model_NN_pf = NN_pf(**attr)
    optimizer, params, opt_state = model.initialize_PF_optimizer(model_NN_pf)
    functions = define_pf_functions(optimizer, model_NN_pf, model.scalers)
    train_NN = True
    if train_NN:
        start_time = time.time()
        params, opt_state = model.train_PF(functions, params, opt_state, epoch_print=50 , patience=8)
        end_time = time.time() 
        execution_time = end_time - start_time  
        print(f"Training the DNN time: {execution_time:.4f} seconds")
        save_NN_model(params, model_NN_pf, path_model)
