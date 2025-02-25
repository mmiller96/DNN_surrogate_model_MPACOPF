from power_grid import true_x_dp, calc_power_flows, decouple_V_angles
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax

def define_pf_functions(optimizer, model_NN_pf, scalers):
    mean_indp = jnp.array(scalers['scaler_X_indp'].mean_)
    std_indp = jnp.array(scalers['scaler_X_indp'].scale_)
    mean_dp = jnp.array(scalers['scaler_X_dp'].mean_)
    std_dp = jnp.array(scalers['scaler_X_dp'].scale_)
    
    def scaling_x_indp(x_indp):
        return (x_indp-mean_indp)/std_indp
    def scaling_x_dp(x_dp):
        return (x_dp-mean_dp)/std_dp
    def inverse_scaling_x_indp(x_indp_scaled):
        return x_indp_scaled * std_indp + mean_indp
    def inverse_scaling_x_dp(x_dp):
        return x_dp*std_dp + mean_dp
    @jit
    def power_flow_scaled(params, x_indp):
        x_indp_scaled = scaling_x_indp(x_indp)
        x_dp_scaled = model_NN_pf.apply({'params': params}, x_indp_scaled)
        return x_dp_scaled
    @jit
    def power_flow(params, x_indp):
        x_indp_scaled = scaling_x_indp(x_indp)
        x_dp_pred_scaled = model_NN_pf.apply({'params': params}, x_indp_scaled)
        x_dp_pred = inverse_scaling_x_dp(x_dp_pred_scaled)
        return x_dp_pred
    
    @jit
    def loss_PF(params, x_indp, x_dp_true):
        x_dp_true_scaled = scaling_x_dp(x_dp_true)
        x_indp_scaled = scaling_x_indp(x_indp)
        x_dp_pred_scaled = model_NN_pf.apply({'params': params}, x_indp_scaled)
        return jnp.mean((x_dp_pred_scaled - x_dp_true_scaled)** 2) 
    
    @jit
    def update_PF(params, opt_state, x_indp, x_dp_true):
        #loss_PF(params, x_PF, y_true)
        grads = grad(loss_PF)(params, x_indp, x_dp_true)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    
    functions = {'power_flow':power_flow, 'loss_PF':loss_PF, 'update_PF':update_PF,
                 'power_flow_scaled':power_flow_scaled, 'scaling_x_dp':scaling_x_dp,
                 'scaling_x_indp':scaling_x_indp, 'inverse_scaling_x_indp':inverse_scaling_x_indp,
                 'inverse_scaling_x_dp':inverse_scaling_x_dp, 'true_x_dp':true_x_dp, 
                 'decouple_V_angles':decouple_V_angles}
    return functions



def define_opf_functions(functions_pf, config, optimizer, params, price):
        # Extract necessary information from config_pf
        idx_RES = config['idx_RES']
        idx_BESS = config['idx_BESS']
        idx_FC = config['idx_FC']
        idx_EL = config['idx_EL']
        num_nodes = config['num_nodes']
        num_lines= config['num_branches']
        num_RES = len(idx_RES)
        num_BESS = len(idx_BESS)
        num_FC = len(idx_FC)
        num_EL = len(idx_EL)
        V_limits = config['V_slack_limits']
        eta_dch, eta_ch = config['eta_dch'], config['eta_ch']
        alpha_EL, beta_FC = config['alpha_EL'], config['beta_FC']
        tan_max = jnp.tan(jnp.arccos(config['cos_phi']))
        S_BESS_rated = jnp.array(config['S_BESS_rated'])[jnp.array(idx_BESS)] if len(idx_BESS) > 0 else jnp.array([])
        S_BESS_capacity = jnp.array(config['S_BESS_capacity'])[jnp.array(idx_BESS)] if len(idx_BESS) > 0 else jnp.array([])
        S_FC_rated = jnp.array(config['S_FC_rated'])[jnp.array(idx_FC)] if len(idx_FC) > 0 else jnp.array([])
        P_EL_rated = jnp.array(config['P_EL_rated'])[jnp.array(idx_EL)] if len(idx_EL) > 0 else jnp.array([])
        G = config['G']
        B = config['B']
        lines = config['lines']
        B_sh = config['B_sh']
        S_line_max = 100.0
        H2_tank = config['hydrogen_tank']
        # Define functions for optimal power flow  
        def insert_array(x, indices):
            new_array = jnp.zeros((x.shape[0], num_nodes))
            new_array = new_array.at[:, indices].set(x)
            return new_array
        
        def compute_SOC_BESS(S_BESS, P_BESS):
            SOC_all = jnp.zeros((S_BESS.shape[0]+1, S_BESS.shape[1]))
            SOC_all = SOC_all.at[0, :].set(0)

            def update_SOC(t, SOC_all):
                S_BESS_t = S_BESS[t-1, :]  # Note the t-1 to align with the shifted SOC array
                P_BESS_t = P_BESS[t-1, :]
                discharge = (SOC_all[t-1, :] - (S_BESS_t / (S_BESS_capacity * eta_dch)))
                charge = (SOC_all[t-1, :] + (S_BESS_t * eta_ch / S_BESS_capacity))
                SOC_t = jnp.where(P_BESS_t >= 0, discharge, charge)
                SOC_all = SOC_all.at[t, :].set(SOC_t)
                return SOC_all
            
            SOC_all = jax.lax.fori_loop(1, S_BESS.shape[0]+1, update_SOC, SOC_all)
            SOC = SOC_all[1:,:]
            return SOC_all, SOC
        
        def compute_HSL(S_FC, P_EL):
            HSL_all = jnp.zeros(S_FC.shape[0]+1)
            HSL_all = HSL_all.at[0].set(0)
            def update_HSL(t, HSL_all):
                S_FC_summed_t = jnp.sum(S_FC[t-1, :])  # Note the t-1 to align with the shifted SOC array
                P_EL_summed_t = jnp.sum(P_EL[t-1, :])
                HSL_t = HSL_all[t-1] - (alpha_EL*P_EL_summed_t + beta_FC*S_FC_summed_t)/H2_tank
                HSL_all = HSL_all.at[t].set(HSL_t)
                return HSL_all
            HSL_all = jax.lax.fori_loop(1, S_FC.shape[0]+1, update_HSL, HSL_all)
            HSL = HSL_all[1:]
            return HSL_all, HSL

        def define_constraint_boundary_vector(vars):
            P_RES, Q_RES, P_BESS, Q_BESS, P_FC, Q_FC, P_EL, V, \
            S_line, S_RES, S_BESS, S_FC, S_RES_avail, SOC_BESS, HSL, S_ext = \
            vars['P_RES'], vars['Q_RES'], vars['P_BESS'], vars['Q_BESS'], \
            vars['P_FC'], vars['Q_FC'], vars['P_EL'], vars['V'], \
            vars['S_line'], vars['S_RES'], vars['S_BESS'], vars['S_FC'], \
            vars['S_RES_avail'], vars['SOC_BESS'], vars['HSL'], vars['S_ext']
            x_lb = jnp.concatenate((0.0*jnp.ones((S_RES.shape[0],1)), jnp.zeros(S_BESS.shape),
                                    jnp.zeros(S_RES.shape), jnp.zeros(S_RES.shape), -P_RES*tan_max,
                                    jnp.zeros(S_BESS.shape), jnp.zeros(S_BESS.shape), -jnp.abs(P_BESS)*tan_max,
                                    jnp.zeros(S_FC.shape), jnp.zeros(S_FC.shape), -P_FC*tan_max,
                                    -jnp.ones(P_EL.shape)*P_EL_rated,
                                    V_limits[0]*jnp.ones(V.shape), jnp.zeros(S_line.shape),
                                    -7.5*jnp.ones((S_RES.shape[0],1))), axis=1)
            x_ineq = jnp.concatenate((HSL.reshape(-1,1), SOC_BESS,  
                                      S_RES, P_RES, Q_RES, 
                                      S_BESS, jnp.abs(P_BESS), Q_BESS, 
                                      S_FC, P_FC, Q_FC, 
                                      P_EL, 
                                      V, S_line,
                                      S_ext.reshape(-1,1)), axis=1)
            x_ub = jnp.concatenate((1*jnp.ones((S_RES.shape[0],1)), jnp.ones(S_BESS.shape),
                                    S_RES_avail, S_RES, P_RES*tan_max,
                                    jnp.ones(S_BESS.shape)*S_BESS_rated, S_BESS, jnp.abs(P_BESS)*tan_max,
                                    jnp.ones(S_FC.shape)*S_FC_rated, S_FC, P_FC*tan_max,
                                    jnp.zeros(P_EL.shape),
                                    V_limits[1]*jnp.ones(V.shape), jnp.ones(S_line.shape)*S_line_max,
                                    7.5*jnp.ones((S_RES.shape[0],1))), axis=1)
            x_LH = jnp.stack(((x_lb-x_ineq), (x_ineq-x_ub)), axis=1)
            return x_LH
        
        def calc_ancillary_variables(vars):
            V, angle = functions_pf['decouple_V_angles'](vars['x_indp'], vars['x_dp'], num_nodes)
            PQ_lines = calc_power_flows(V, angle, lines, B, G, B_sh)
            vars['P_line'] = PQ_lines[:,:num_lines]
            vars['Q_line'] = PQ_lines[:,num_lines:]
            S_line = jnp.abs(vars['P_line']**2 + vars['Q_line']**2)
            S_RES = jnp.sqrt((vars['P_RES']**2 + vars['Q_RES']**2)+1e-20)
            S_BESS = jnp.sqrt((vars['P_BESS']**2 + vars['Q_BESS']**2)+1e-20)
            S_FC = jnp.sqrt((vars['P_FC']**2 + vars['Q_FC']**2)+1e-20)
            S_RES_avail = vars['x_fix'][:,-num_RES:]
            SOC_BESS_all, SOC_BESS = compute_SOC_BESS(S_BESS, vars['P_BESS'])
            HSL_all, HSL = compute_HSL(S_FC, vars['P_EL'])
            vars['P_bus_slack'] = vars['x_dp'][:,-2]
            vars['Q_bus_slack'] = vars['x_dp'][:,-1]
            vars['P_ext'] = vars['P_bus_slack'] - vars['P_DER_slack_summed']
            vars['Q_ext'] = vars['Q_bus_slack'] - vars['Q_DER_slack_summed']
            vars['S_ext'] = jnp.sqrt(vars['P_ext']**2 + vars['Q_ext']**2)
            vars.update({
                'S_line': S_line, 'S_RES': S_RES, 'S_BESS': S_BESS, 'S_FC': S_FC,
                'S_RES_avail': S_RES_avail, 'SOC_BESS_all': SOC_BESS_all, 'HSL_all': HSL_all,
                'SOC_BESS': SOC_BESS, 'HSL': HSL, 'V': V, 'angle':angle
            })
            return vars
        
        def define_inputs(vars):
            """
            Defines the input for the power flow model based on the decision variables and fixed parameters.
            """
            P_L_all = -vars['x_fix'][:, :num_nodes]
            Q_L_all = -vars['x_fix'][:, num_nodes: 2*num_nodes]
        
            P_RES_all = insert_array(vars['P_RES'], idx_RES)
            Q_RES_all = insert_array(vars['Q_RES'], idx_RES)
            P_BESS_all = insert_array(vars['P_BESS'], idx_BESS)
            Q_BESS_all = insert_array(vars['Q_BESS'], idx_BESS)
            P_FC_all = insert_array(vars['P_FC'], idx_FC)
            Q_FC_all = insert_array(vars['Q_FC'], idx_FC)
            P_EL_all = insert_array(vars['P_EL'], idx_EL)

            PQ_RES = jnp.concatenate((P_RES_all[:,1:], Q_RES_all[:,1:]), axis=1)
            PQ_BESS = jnp.concatenate((P_BESS_all[:,1:], Q_BESS_all[:,1:]), axis=1)
            PQ_FC = jnp.concatenate((P_FC_all[:,1:], Q_FC_all[:,1:]), axis=1)
            PQ_EL = jnp.concatenate((P_EL_all[:,1:], jnp.zeros_like(P_EL_all[:,1:])), axis=1)
            PQ_L = jnp.concatenate((P_L_all[:,1:], Q_L_all[:,1:]), axis=1)
            PQ_bus = PQ_RES + PQ_FC + PQ_BESS + PQ_L + PQ_EL
            
            V_slack = vars['V_slack'].reshape(-1, 1)
            angle_slack = jnp.zeros_like(vars['V_slack'].reshape(-1, 1))
            vars['x_indp'] = jnp.concatenate((PQ_bus, V_slack, angle_slack), axis=1)
            vars['P_DER_slack_summed'] = P_RES_all[:,0] + P_BESS_all[:,0] + P_FC_all[:,0] + P_L_all[:,0] + P_EL_all[:,0]
            vars['Q_DER_slack_summed'] = Q_RES_all[:,0] + Q_BESS_all[:,0] + Q_FC_all[:,0] + Q_L_all[:,0]
            return vars

        def decouple_x_pf(x_pf):
            """
            Decouples decision variables into individual components.
            """
            return {
                'P_RES': x_pf[:, :num_RES],
                'Q_RES': x_pf[:, num_RES:2 * num_RES],
                'P_BESS': x_pf[:, 2 * num_RES:2 * num_RES + num_BESS],
                'Q_BESS': x_pf[:, 2 * num_RES + num_BESS:2 * num_RES + 2 * num_BESS],
                'P_FC': x_pf[:, 2 * num_RES + 2 * num_BESS:2 * num_RES + 2 * num_BESS + num_FC],
                'Q_FC': x_pf[:, 2 * num_RES + 2 * num_BESS + num_FC:2 * num_RES + 2 * num_BESS + 2 * num_FC],
                'P_EL': x_pf[:, -(num_EL + 1):-1],
                'V_slack': x_pf[:, -1]
            }

        def calc_all_variables(x_pf, x_fix):
            """
            Calculates all variables needed for constraint evaluation and the objective function of the OPF.
            """
            vars = decouple_x_pf(x_pf)
            vars['x_fix'] = x_fix
            vars = define_inputs(vars)
            vars['x_dp'] = functions_pf['power_flow'](params, vars['x_indp'])
            vars = calc_ancillary_variables(vars)
            return vars

        @jit
        def loss_OPF_seperated(x_pf, x_fix, mu, rho):
            vars = calc_all_variables(x_pf, x_fix)
            x_LH = define_constraint_boundary_vector(vars)
            penalty_linear = mu*jnp.maximum(x_LH,0)
            penalty_quadratic = (rho/2)*jnp.maximum(x_LH,0)**2
            obj_prices = jnp.where(vars['P_ext'] >= 0, vars['S_ext'] * price, vars['P_ext'] * price)
            return obj_prices, penalty_linear, penalty_quadratic
        
        @jit
        def loss_OPF(x_pf, x_fix, mu, rho):
            obj_prices, penalty_linear, penalty_quadratic = loss_OPF_seperated(x_pf, x_fix, mu, rho)
            obj_price = jnp.mean(obj_prices)
            penalty_loss_linear = jnp.mean(penalty_linear)
            penalty_loss_quadratic = jnp.mean(penalty_quadratic)
            return obj_price + penalty_loss_linear + penalty_loss_quadratic
            
        @jit
        def update_OPF(opt_state, x_pf, x_fix, mu, rho):
            grads = grad(loss_OPF, argnums=0)(x_pf, x_fix, mu, rho)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            x_pf = optax.apply_updates(x_pf, updates)
            return x_pf, new_opt_state
        
        functions_opf = {'loss_OPF':loss_OPF, 
                         'loss_OPF_seperated':loss_OPF_seperated,
                         'update_OPF':update_OPF, 
                         'calc_all_variables':calc_all_variables,
                         'define_constraint_boundary_vector': define_constraint_boundary_vector}
        functions_opf.update(functions_pf)
        return functions_opf