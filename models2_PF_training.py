import jax.numpy as jnp
from jax import random
import optax
from loader import Loader_PF

class PowerFlowModel:
    def __init__(self, config):    
        self.config = config
        self.key = random.PRNGKey(self.config['random_state'])
        self.loader = Loader_PF(config)
        self.data, self.scalers = self.loader.initialize_data()
        self.input_size = self.data['X_indp_train'].shape[1]
        self.output_size = self.data['X_dp_train'].shape[1]

    def initialize_PF_optimizer(self, model_NN_pf):
        optimizer = optax.adam(learning_rate=self.config['lr'])
        params = model_NN_pf.init(self.key, jnp.ones([1, self.input_size]))['params']
        opt_state = optimizer.init(params)
        return optimizer, params, opt_state
    
    def train_PF(self, functions, params, opt_state, epoch_print, patience=10):
        best_loss = float('inf')  # Initialize best loss as infinity
        epochs_no_improve = 0     # Counter for epochs with no improvement
        best_params = params      # Store the best model parameters
        best_opt_state = opt_state
        for epoch in range(self.config['num_epochs']):
            train_loader = self.loader.create_data_loader(self.data['X_indp_train'], self.data['X_dp_train'], self.loader.batch_size)
            for X_indp_train_batch, X_dp_train_batch in train_loader:
                params, opt_state = functions['update_PF'](params, opt_state, X_indp_train_batch, X_dp_train_batch)
            if epoch % epoch_print == 0:  # Print loss every 10 epochs
                X_indp_val = self.data['X_indp_val']
                X_dp_val = self.data['X_dp_val']
                loss = functions['loss_PF'](params, X_indp_val, X_dp_val)
                print(f'Epoch {epoch}, Val: {loss}')
                # Early stopping logic
                if loss < best_loss:
                    best_loss = loss
                    best_params = params
                    best_opt_state = opt_state
                    epochs_no_improve = 0  # Reset the counter if improvement
                else:
                    epochs_no_improve += 1  # Increment if no improvement
                    
                if epochs_no_improve >= patience:  # Stop if patience limit is reached
                    print(f'Early stopping at epoch {epoch} with best loss {best_loss}')
                    break
        return best_params, best_opt_state