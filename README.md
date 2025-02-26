preprocess.py – Data Preparation: Prepares the input data and system model for the 33-bus distribution network. This script processes grid configuration and time-series data (loads, generation profiles, prices) needed for subsequent steps. It outputs normalized datasets and parameters to be used by the following scripts.

main1_PF_dataset.py – Power Flow Dataset Generation: Uses the prepared model to run numerous AC power flow simulations (via pandapower) under random load/generation conditions. It generates a large dataset of input-output pairs (grid injections -> resulting voltages/angles) for training the surrogate model​.

main2_PF_training.py – Surrogate Model Training: Trains a deep neural network on the dataset from step 2. The DNN learns to predict power flow outcomes (bus voltages, angles, etc.) from grid injection inputs. The trained model (surrogate) is saved for use in the optimization phase​.
​
main3_OPF_optimizer.py – MP-ACOPF Optimization with Surrogate: Solves the multi-period optimal power flow problem (e.g. a one-month horizon) using the DNN surrogate in place of iterative power flow calculations​. It employs an Augmented Lagrangian Method (ALM) to enforce power flow constraints and optimize control variables (generator dispatch, storage usage, external grid exchange) over all time steps​. This yields the optimal schedule of generation and storage operation minimizing total cost while respecting network limits.

results_plots.py – Results Analysis and Visualization: After the optimization, this script loads the saved surrogate model and optimization results to produce plots and figures. These include error analysis of the surrogate model and time-series of the optimized grid operation, mirroring the key results from the associated research paper.
