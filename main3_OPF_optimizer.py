from models3_OPF_Optimizer import OptimalPowerFlowOptimizer
import pdb
from file_handler import save_pickle
from functions import define_opf_functions, define_pf_functions
import jax
import time
import os

if __name__== '__main__':
    jax.config.update('jax_xla_backend', 'cpu')
    print(jax.default_backend())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(script_dir, "data", "ieee33dataset.h5")
    path_model = os.path.join(script_dir, "models", "ieee33_DNN_best.h5")
    path_result = os.path.join(script_dir, "results", "ieee33results_April2.h5")
    name = 'ieee33'

    config = {'path_data': path_dataset, 'lr':1e-3 , 'rho': 10000.0, 'penalty_factor_ineq':20000.0, 'epsilon': 1e-4, 
              'max_inner_iterations':100000, 'max_outer_iterations':100, 'iterations_print': 500, 'patience':3}
    model = OptimalPowerFlowOptimizer(config)
    model_NN_pf, params, attr = model.load_pf_model(path_model)
    x_opf_fix_df, scalers, config_pf, net, price = model.load_pf_dataset()
    opt_state, x_dv = model.initialize_opt_state(x_opf_fix_df, config_pf)
    functions_pf = define_pf_functions(model.optimizer, model_NN_pf, scalers)
    functions_opf = define_opf_functions(functions_pf, config_pf, model.optimizer, params, price)

    start_time = time.time()
    results = model.run_MPACOPF(x_opf_fix_df, x_dv, functions_opf, opt_state)
    end_time = time.time() 
    execution_time = end_time - start_time  
    print(f"MP-ACOPF optimization time: {execution_time:.4f} seconds")
    save_pickle(path_result, results, name)