from models_NN import NN_pf
import numpy as np
import pdb
from jax import random, device_put
import jax.numpy as jnp
import optax
from file_handler import load_NN_model, load_power_flow_dataset
import matplotlib.pyplot as plt 
from utils import transform_to_numpy

class OptimalPowerFlowOptimizer:
    def __init__(self, config):
        self.config = config
        self.key = random.PRNGKey(41)
        self.optimizer = optax.adam(learning_rate=self.config['lr'])
        
    def load_pf_model(self, path_PF_model):
        params, attr = load_NN_model(path_PF_model)
        model_NN_pf = NN_pf(**attr)
        return model_NN_pf, params, attr
    
    def load_pf_dataset(self):
        _, _, scaler_X_indp, scaler_X_dp, config_pf, X_opf_fix, data_df, net = load_power_flow_dataset(self.config['path_data'])
        X_opf_fix = X_opf_fix.iloc[90*24:120*24]   # April
        data_df = data_df.iloc[90*24:120*24]
        price = data_df['Deutschland/Luxemburg [€/MWh] Originalauflösungen'].values
        self.data_df = data_df
        scalers = {
            'scaler_X_indp': scaler_X_indp,
            'scaler_X_dp': scaler_X_dp
        }
        return X_opf_fix, scalers, config_pf, net, price

    def initialize_opt_state(self, x_opf_fix_df, config_pf):
        num_RES = len(config_pf['idx_RES'])
        num_BESS = len(config_pf['idx_BESS'])
        num_FC = len(config_pf['idx_FC'])
        num_EL = len(config_pf['idx_EL'])
        S_RES_avail = x_opf_fix_df.iloc[:,-num_RES:].values
        P_RES = S_RES_avail.copy()
        Q_RES = jnp.zeros_like(P_RES)
        P_BESS = jnp.zeros((len(P_RES), num_BESS))
        Q_BESS = jnp.zeros((len(Q_RES), num_BESS))
        P_FC = jnp.zeros((len(P_RES), num_FC))
        Q_FC = jnp.zeros((len(Q_RES), num_FC))
        P_EL = jnp.zeros((len(P_RES), num_EL))
        V_slack = jnp.ones((P_RES.shape[0], 1))
        x_pf = jnp.concatenate((P_RES, Q_RES, P_BESS, Q_BESS, P_FC, Q_FC, P_EL, V_slack), axis=1)
        opt_state = self.optimizer.init(x_pf)
        return opt_state, x_pf
    
    def run_MPACOPF(self, x_opf_fix_df, x_pf, functions, opt_state):
        x_opf_fix_values = device_put(x_opf_fix_df.values)

        # Initialize variables
        vars = functions['calc_all_variables'](x_pf, x_opf_fix_values)
        x_LH = functions['define_constraint_boundary_vector'](vars)
        mu = jnp.zeros_like(x_LH)  # Initialize Lagrange multipliers
        rho = self.config['rho']
        best_x_pf = x_pf                  # Initialize best decision variables
        max_outer_iterations = self.config['max_outer_iterations']
        for outer_iter in range(self.config['max_outer_iterations']):
            print(f'\n--- Outer Iteration {outer_iter + 1}/{max_outer_iterations} ---')
            print(f'Penalty parameter rho: {jnp.mean(rho)}')
            opt_state = self.optimizer.init(best_x_pf)
            best_inner_loss = float('inf')
            iterations_no_improve = 0
            for j in range(self.config['max_inner_iterations']):
                functions['loss_OPF_seperated'](x_pf, x_opf_fix_values, mu, rho)
                x_pf, opt_state = functions['update_OPF'](opt_state, x_pf, x_opf_fix_values, mu, rho)
                if j % self.config['iterations_print'] == 0:
                    loss_obj, penalty_linear, penalty_quadratic = functions['loss_OPF_seperated'](x_pf, x_opf_fix_values, mu, rho)
                    total_loss = jnp.mean(loss_obj) + jnp.mean(penalty_linear) + jnp.mean(penalty_quadratic)
                    print(f'Epoch {j}, Total: {total_loss.item():.6f}, '
                          f'Objective: {jnp.mean(loss_obj).item():.6f}, '
                          f'Linear Penalty: {jnp.mean(penalty_linear).item():.6f}, '
                          f'Quadratic Penalty: {jnp.mean(penalty_quadratic).item():.6f}, '
                          f'rho: {jnp.mean(rho)}, mu_mean: {jnp.mean(mu).item():.6f}')
                    # Early stopping logic
                    if total_loss < best_inner_loss:
                        best_inner_loss = total_loss
                        best_x_pf = x_pf  # Update best decision variables
                        iterations_no_improve = 0  # Reset counter
                    else:
                        iterations_no_improve += 1  # Increment counter
                    if iterations_no_improve >= self.config['patience']:
                        print(f'Early stopping at inner iteration {j} with best loss {best_inner_loss:.6f}')
                        break

            # After inner loop, update multipliers mu
            vars = functions['calc_all_variables'](x_pf, x_opf_fix_values)
            x_LH = functions['define_constraint_boundary_vector'](vars)
            x_pf = best_x_pf
            # plt.plot(vars['HSL_all'])
            # path_save = 'results/pictures/test/' + str(outer_iter) + '_HSL.pdf'
            # plt.savefig(path_save)
            # plt.close()
            # plt.plot(vars['SOC_BESS_all'])
            # path_save = 'results/pictures/test/' + str(outer_iter) + '_SOC.pdf'
            # plt.savefig(path_save)
            # plt.close()
            mu = mu + rho * jnp.maximum(x_LH, 0)
            max_constraint_violation = jnp.max(jnp.maximum(0, x_LH)).item()

            print(f'Max constraint violation after outer iteration {outer_iter + 1}: {max_constraint_violation:.6e}')
            # Final stopping criterion
            if max_constraint_violation <= self.config['epsilon']:
                print(f'Constraints satisfied within tolerance {max_constraint_violation:.6e}')
                break  # Stop outer loop if constraints are satisfied
            rho *= 5  # Increase rho to penalize constraint violations more

        vars = functions['calc_all_variables'](best_x_pf, x_opf_fix_values)
        vars['loss_obj'], vars['loss_linear'], vars['loss_quadratic'] = functions['loss_OPF_seperated'](best_x_pf, x_opf_fix_values,mu,rho)
        vars = transform_to_numpy(vars)
        vars['data_df'] = self.data_df
        return vars
