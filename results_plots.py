
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch
from file_handler import load_pickle, load_power_flow_dataset
import os


if __name__ == '__main__':
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    # Optionally specify the LaTeX preamble to include specific packages
    "text.latex.preamble": r"\usepackage{amsmath}",
    })
    size = 40
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(script_dir, "data", "ieee33dataset.h5")
    path_result = os.path.join(script_dir, "results", "ieee33results_April.h5")
    name = 'ieee33'

    _, _, _, _, config, X_opf_fix, data_df, net = load_power_flow_dataset(path_dataset)
    results = load_pickle(path_result, name)
    loss_obj = results['loss_obj']
    data_df = results['data_df']
    num_nodes = config['num_nodes']

    # x and y ticks for whole April
    prices = data_df['Deutschland/Luxemburg [€/MWh] Originalauflösungen'].values
    time_axis = np.arange(len(prices))
    days = np.arange(0, 30, 7)  
    days_positions = days * 24  # Positions in time_axis (every 24 hours)

    # x and y ticks for second week in April
    prices_week = prices[24*7-1:24*14]
    time_axis_week = time_axis[24*7-1:24*14]
    days_week = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    days_week_position = time_axis_week[::24]
    
    # if true, create the plots
    plot_voltages = True
    plot_prices = True
    plot_cum_price = True
    plot_WT = True
    plot_PV = True
    plot_S_RES = True
    plot_P_stacked_area = True
    plot_Q_stacked_area = True
    plot_SOCs = True
    
    if plot_voltages:  
        voltages = results['V'][24*7-1:24*14]
        plt.figure(figsize=(16, 8))
        plt.plot(time_axis_week, voltages[:, 0], color='blue', linestyle='-', linewidth=2, zorder=10, label='Bus 1 (Slack)', drawstyle='steps-mid')
        for bus_idx in range(1, voltages.shape[1]):
            plt.plot(time_axis_week, voltages[:, bus_idx], color='lightblue', linestyle='-', linewidth=1, alpha=0.6, drawstyle='steps-mid')
        plt.axhline(y=0.95, color='black', linestyle='--', linewidth=2)
        plt.axhline(y=1.05, color='black', linestyle='--', linewidth=2)
        plt.xlim(time_axis_week[0], time_axis_week[-1])
        plt.ylim(0.935, 1.065) 
        plt.xticks(days_week_position, labels=days_week, fontsize=size)
        plt.tick_params(axis='y', labelsize=size) 
        plt.plot([], [], color='lightblue', linestyle='-', linewidth=2, alpha=1.0, label='Bus 2-33')
        legend = plt.legend(
            loc='lower left', 
            bbox_to_anchor=(-0.02, -0.042),  
            fontsize=size, 
            framealpha=0.7, 
            facecolor='white', 
            edgecolor='black', 
            fancybox=True,
        )
        plt.xlabel('Days in April', fontsize=size)
        plt.ylabel(r'$\text{Voltage} [\text{p.u.}]$', fontsize=size)
        plt.tight_layout()
        plt.grid(True)
        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Voltages_April.pdf")
        plt.savefig(path_save, format='pdf')
        plt.close()
        voltages = results['V']
        time_axis = np.arange(len(voltages))
        

    if plot_cum_price:
        cumulative_cost = np.cumsum(loss_obj)/1000
        fig, ax = plt.subplots(figsize=(16, 8))
        color = 'tab:red'
        ax.plot(time_axis, cumulative_cost, color=color, linestyle='-', linewidth=4, label=r'Cumulative Cost')
        ax.set_xlabel('Days in April', fontsize=size)
        ax.set_ylabel('Cumulative Cost [k€]', fontsize=size)
        ax.tick_params(axis='y', labelsize=size)
        ax.grid(True)
        # Adjust x-axis tick labels
        ax.set_xticks(days_positions)
        ax.set_xticklabels(days, fontsize=size)
        ax.set_xlim(time_axis[0], time_axis[-1])
        # Save the plot
        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Cumulative_Cost_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save, bbox_inches='tight')
        plt.close()

    if plot_prices:
        fig, ax = plt.subplots(figsize=(16, 8))
        color = 'tab:blue'
        ax.set_xlabel('Days in April', fontsize=size)
        ax.set_ylabel('Electricity Price [€/MWh]', fontsize=size)
        ax.plot(time_axis, prices, color=color, linestyle='-', linewidth=4, label='Electricity Prices')
        ax.tick_params(axis='y',  labelsize=size)
        ax.set_xticks(days_positions) 
        ax.set_xticklabels(days, fontsize=size)  
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.grid(True)

        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Price_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save)
        plt.close()

    if plot_WT:
        wind_speed = data_df['wind_speed_100m (km/h)'].values
        eta_WT = data_df['eta_WT'].values
        # Plot eta_WT and wind speed with two different y-axes
        fig, ax1 = plt.subplots(figsize=(16, 8))

        # Plot wind speed on the left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Days in April', fontsize=size)
        ax1.plot(time_axis, wind_speed, color=color, linestyle='-', linewidth=2, label='Wind Speed')
        ax1.tick_params(axis='y', labelcolor=color, labelsize=size)
        ax1.grid(True)

        # Create another y-axis for eta_WT
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.plot(time_axis, eta_WT, color=color, linestyle='--', linewidth=2, label='eta_WT')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=size)

        legend_elements = [
        Patch(facecolor='tab:blue', edgecolor='blue', label=r'$\text{Wind speed} (\frac{m}{s})$'),
        Patch(facecolor='tab:green', edgecolor='green', label=r'$\epsilon_{WT}$')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=size)
        ax1.set_xticks(days_positions)  # Use the specified day positions
        ax1.set_xticklabels(days, fontsize=size)  # Set tick labels and font size explicitly
        ax1.set_xlim(time_axis[0], time_axis[-1])
        ax1.set_ylim(0, 50)
        ax2.set_ylim(0, 1.05) 

        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Wind_speed_factor_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save)
        plt.close()

    if plot_PV:
        # Extract solar irradiance and eta_PV from the data_df
        solar_irradiance = data_df['shortwave_radiation (W/m²)'].values
        eta_PV = data_df['eta_PV'].values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        color = 'tab:blue'
        ax1.set_xlabel('Days in April', fontsize=size)
        ax1.plot(time_axis, solar_irradiance, color=color, linestyle='-', linewidth=2, label='Solar Irradiance')
        ax1.tick_params(axis='y', labelcolor=color, labelsize=size)
        ax1.grid(True)
        ax1.set_xticks(days_positions)
        ax1.set_xticklabels(days, fontsize=size)
        ax1.set_xlim(time_axis[0], time_axis[-1])

        legend_elements = [
        Patch(facecolor='tab:blue', edgecolor='blue', label=r'$\text{Solar irradiance} (\frac{W}{m^2})$'),
        Patch(facecolor='tab:green', edgecolor='green', label=r'$\epsilon_{PV}$')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=size)
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.plot(time_axis, eta_PV, color=color, linestyle='--', linewidth=2, label=r'$\epsilon{PV}$')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=size)

        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Solar_PV_Efficiency_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save)
        plt.close()

    if plot_S_RES:
        S_RES_avail_columns = X_opf_fix.iloc[:, 2*num_nodes:].columns
        S_RES_avail_labels = [rf"$S_{{\text{{RES}},{idx}}}$" for idx in config['idx_RES']]
        S_RES_labels = [rf"$S_{{\text{{RES}},{idx}}}$" for idx in config['idx_RES']]
        S_RES = results['S_RES']
        S_RES_avail = results['S_RES_avail']
        S_RES_avail_total = np.sum(S_RES_avail, axis=1)
        cmap = colormaps["Reds"]
        colors = [cmap(i) for i in np.linspace(0.3, 1.0, S_RES.shape[1])]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.stackplot(
            time_axis, 
            S_RES.T, 
            labels=S_RES_labels, 
            colors=colors[:S_RES.shape[1]],  
            alpha=0.8,
        )
        ax.plot(
            time_axis, 
            S_RES_avail_total, 
            linestyle='-', 
            linewidth=1.5, 
            color='black', 
            label=r"$\sum S_{\text{RES},j}^{\text{avail}}$",
        )
        # Add labels, ticks, and legend
        ax.set_xlabel('Days in April', fontsize=size)
        ax.set_ylabel('Apparent Power [MVA]', fontsize=size)
        ax.tick_params(axis='y', labelsize=size)  # Adjust y-tick font size
        ax.legend(loc='upper left', fontsize=size*0.7, ncol=2)
        ax.grid(True)

        # Set x-ticks for days in April
        ax.set_xticks(days_positions)
        ax.set_xticklabels(days, fontsize=size)
        ax.set_xlim(time_axis[0], time_axis[-1])
        # Save and close the plot
        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Apparent_Power_Stacked_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save)
        plt.close()


    if plot_P_stacked_area:
        P_FC = results['P_FC'][24*7-1:24*14]  
        P_EL = results['P_EL'][24*7-1:24*14]
        P_BESS = results['P_BESS'][24*7-1:24*14] 
        P_RES = results['P_RES'][24*7-1:24*14]
        P_L = -results['x_fix'][24*7-1:24*14, :num_nodes].sum(axis=1) 
        P_ext = results['P_ext'][24*7-1:24*14]

        P_FC_sum = P_FC.sum(axis=1)  
        P_BESS_sum = P_BESS.sum(axis=1) 
        P_RES_sum = P_RES.sum(axis=1) 
        P_EL_sum = P_EL.sum(axis=1) 
        fig, ax = plt.subplots(figsize=(16, 8))
        colors = {
            'P_FC': 'teal',
            'P_BESS': 'tab:orange',
            'P_EL': 'tab:green',
            'P_RES': 'tab:red',
            'P_L': 'gray',
            'P_ext': 'tab:blue'
        }
        cumulative_positive = np.zeros_like(time_axis_week, dtype=np.float64)
        positive_bess = np.maximum(P_BESS_sum, 0)
        negative_bess = np.minimum(P_BESS_sum, 0)  # Negative values for batteries
        ax.fill_between(
            time_axis_week, 
            cumulative_positive, 
            cumulative_positive + positive_bess, 
            color=colors['P_BESS'], 
            alpha=0.8, 
            label=r'$\sum \text{P}_{\text{BESS},j}$',
            step='mid'
        )
        cumulative_positive += positive_bess
        ax.fill_between(
            time_axis_week, 
            cumulative_positive, 
            cumulative_positive + P_FC_sum, 
            color=colors['P_FC'], 
            alpha=0.8, 
            label=r'$\sum \text{P}_{\text{FC},j}$',
            step='mid'
        )
        cumulative_positive += P_FC_sum      
        ax.fill_between(
            time_axis_week, 
            cumulative_positive, 
            cumulative_positive + P_RES_sum, 
            color=colors['P_RES'], 
            alpha=0.8, 
            label=r'$\sum \text{P}_{\text{RES},j}$',
            step='mid'
        )
        cumulative_positive += P_RES_sum
        cumulative_negative = np.zeros_like(time_axis_week, dtype=np.float64)
        ax.fill_between(
            time_axis_week, 
            cumulative_negative, 
            cumulative_negative + negative_bess, 
            color=colors['P_BESS'], 
            alpha=0.8, 
            step='mid'
        )
        cumulative_negative += negative_bess
        ax.fill_between(
            time_axis_week, 
            cumulative_negative, 
            cumulative_negative + P_EL_sum, 
            color=colors['P_EL'], 
            alpha=0.8, 
            label=r'$\text{P}_{\text{EL},1}$',
            step='mid'
        )
        cumulative_negative += P_EL_sum
        ax.fill_between(
            time_axis_week, 
            cumulative_negative, 
            cumulative_negative + P_L, 
            color=colors['P_L'], 
            alpha=0.8, 
            label=r'$\sum \text{P}_{\text{L},j}$',
            step='mid'
        )
        cumulative_negative += P_L
        positive_ext = np.maximum(P_ext, 0)
        negative_ext = np.minimum(P_ext, 0)
        ax.fill_between(
            time_axis_week, 
            cumulative_positive, 
            cumulative_positive + positive_ext, 
            color=colors['P_ext'], 
            alpha=0.8, 
            label=r'$\text{P}_{\text{ext}}$',
            step='mid'
        )
        cumulative_positive += positive_ext 
        ax.fill_between(
            time_axis_week, 
            cumulative_negative, 
            cumulative_negative + negative_ext, 
            color=colors['P_ext'], 
            alpha=0.8, 
            label=r'$\text{P}_{\text{ext}}^{-}$',
            step='mid'
        )
        cumulative_negative += negative_ext 
        cumulative_negative += P_L
        handles, labels = ax.get_legend_handles_labels()
        first_row = [handles[0], handles[1], handles[2]]  # Battery, Fuel Cell, RES
        second_row = [handles[3], handles[4], handles[5]]  # Electrolyzer, Load, Ext

        ax.legend(
            first_row + second_row,
            [labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]],
            loc='upper center',
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,  
            fontsize=size * 0.9
        )
        ax.set_xlabel('Days in April', fontsize=size)
        ax.set_ylabel('Active Power [MW]', fontsize=size)
        ax.grid(True)
        # Add x-ticks for days in April
        ax.set_xticks(days_week_position)
        ax.set_xticklabels(days_week, fontsize=size)
        # Adjust y-tick font size
        ax.tick_params(axis='y', labelsize=size)
        ax.set_xlim(time_axis_week[0], time_axis_week[-1])
        ax.set_ylim(-10, 15)
        # Save and show the plot
        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Active_Power_Stacked_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save, bbox_inches='tight')
        plt.close()

    if plot_Q_stacked_area:
        Q_FC = results['Q_FC'][24*7-1:24*14] 
        Q_BESS = results['Q_BESS'][24*7-1:24*14] 
        Q_RES = results['Q_RES'][24*7-1:24*14]
        Q_L = -results['x_fix'][24*7-1:24*14, num_nodes:2*num_nodes].sum(axis=1) 
        Q_ext = results['Q_ext'][24*7-1:24*14]

        Q_FC_sum = Q_FC.sum(axis=1) 
        Q_BESS_sum = Q_BESS.sum(axis=1) 
        Q_RES_sum = Q_RES.sum(axis=1)
        # Create a stacked area plot
        fig, ax = plt.subplots(figsize=(16, 8))

        colors = {
        'Q_FC': 'teal',
        'Q_BESS': 'tab:orange',
        'Q_RES': 'tab:red',
        'Q_L': 'gray',
        'Q_ext': 'tab:blue'
        }
        cumulative_positive = np.zeros_like(time_axis_week, dtype=np.float64)
        positive_bess = np.maximum(Q_BESS_sum, 0)
        negative_bess = np.minimum(Q_BESS_sum, 0) 
        ax.fill_between(
        time_axis_week, 
        cumulative_positive, 
        cumulative_positive + positive_bess, 
        color=colors['Q_BESS'], 
        alpha=0.8, 
        label=r'$\sum \text{Q}_{\text{BESS},j}$',
        step='mid'
        )
        cumulative_positive += positive_bess
        ax.fill_between(
        time_axis_week, 
        cumulative_positive, 
        cumulative_positive + Q_FC_sum, 
        color=colors['Q_FC'], 
        alpha=0.8, 
        label=r'$\sum \text{Q}_{\text{FC},j}$',
        step='mid'
        )
        cumulative_positive += Q_FC_sum      
        ax.fill_between(
        time_axis_week, 
        cumulative_positive, 
        cumulative_positive + Q_RES_sum, 
        color=colors['Q_RES'], 
        alpha=0.8, 
        label=r'$\sum \text{Q}_{\text{RES},j}$',
        step='mid'
        )
        cumulative_positive += Q_RES_sum
        cumulative_negative = np.zeros_like(time_axis_week, dtype=np.float64)
        ax.fill_between(
        time_axis_week, 
        cumulative_negative, 
        cumulative_negative + negative_bess, 
        color=colors['Q_BESS'], 
        alpha=0.8, 
        step='mid'
        )
        cumulative_negative += negative_bess
        ax.fill_between(
        time_axis_week, 
        cumulative_negative, 
        cumulative_negative + Q_L, 
        color=colors['Q_L'], 
        alpha=0.8, 
        label=r'$\sum \text{Q}_{\text{L},j}$',
        step='mid'
        )
        cumulative_negative += Q_L
        negative_ext = np.minimum(Q_ext, 0)
        positive_ext = np.maximum(Q_ext, 0)
        ax.fill_between(
        time_axis_week, 
        cumulative_positive, 
        cumulative_positive + positive_ext, 
        color=colors['Q_ext'], 
        alpha=0.8, 
        label=r'$\text{Q}_{\text{ext}}$',
        step='mid'
        )
        cumulative_positive += positive_ext
        ax.fill_between(
        time_axis_week, 
        cumulative_negative, 
        cumulative_negative + negative_ext, 
        color=colors['Q_ext'], 
        alpha=0.8, 
        label=r'$\text{Q}_{\text{ext}}^{-}$',
        step='mid'
        )
        cumulative_negative += negative_ext  
        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        first_row = [handles[0], handles[1], handles[2]]  # Battery, Fuel Cell, RES
        second_row = [handles[3], handles[4]]  # Load, External grid
        ax.legend(
        first_row + second_row,
        [labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,  # 3 columns for the first row
        fontsize=size * 0.9
        )
        # Add labels, legend, and grid
        ax.set_xlabel('Days in April', fontsize=size)
        ax.set_ylabel('Reactive Power [MVAr]', fontsize=size)  # Changed to MVAr
        ax.grid(True)

        # Add x-ticks for days in April
        ax.set_xticks(days_week_position)
        ax.set_xticklabels(days_week, fontsize=size)

        # Adjust y-tick font size
        ax.tick_params(axis='y', labelsize=size)
        ax.set_xlim(time_axis_week[0], time_axis_week[-1])
        ax.set_ylim(-3, 4)

        # Save the plot
        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Reactive_Power_Stacked_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save, bbox_inches='tight')
        plt.close()



    if plot_SOCs:
        # # Extract state of charge for BESS (SOC_BESS) and hydrogen storage (SOC_H2)
        SOC_BESS = results['SOC_BESS']
        HSL = results['HSL']
        # Energy capacities
        battery_capacities = [5, 5, 5]  # MWh for each battery
        hydrogen_tank_capacity = 20  # MWh

        # Transform SOCs to energy
        battery_energies = SOC_BESS * np.array(battery_capacities)  # Shape: (time, 3)
        hydrogen_energy = HSL * hydrogen_tank_capacity  # Shape: (time,)

        battery_labels = [f'Battery {i + 1}' for i in range(SOC_BESS.shape[1])]
        colors = ['lightcoral', 'indianred', 'brown']  # Different shades of red for batteries

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot batteries in stacked area
        ax.stackplot(
            time_axis, 
            battery_energies.T, 
            labels=battery_labels, 
            colors=colors, 
            alpha=0.8,
        )

        # Plot hydrogen energy as the top layer
        ax.plot(
            time_axis, 
            hydrogen_energy + np.sum(battery_energies, axis=1), 
            color='tab:blue', 
            linewidth=2.5, 
            label='Hydrogen Tank'
        )
        ax.fill_between(
            time_axis, 
            np.sum(battery_energies, axis=1), 
            np.sum(battery_energies, axis=1) + hydrogen_energy, 
            color='tab:blue', 
            alpha=0.5
        )

        # Add labels and legend
        ax.set_xlabel('Days in April', fontsize=size)
        ax.set_ylabel('Total Energy [MWh]', fontsize=size)

        ax.set_xticks(days_positions)
        ax.set_xticklabels(days, fontsize=size)
        ax.tick_params(axis='y', labelsize=size)
        ax.axhline(y=35, color='black', linestyle='--', linewidth=2)
        ax.text(120, 37.2, 'Maximum capacity', fontsize=size, color='black', ha='center', va='center')

        # Add grid, legend, and limits
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=size*0.8)
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(0, hydrogen_tank_capacity + sum(battery_capacities)+12)  # Full system capacity

        # Save the plot
        path_save = os.path.join(script_dir, "results", "pictures", "documentation", "Energy_Stacked_April.pdf")
        plt.tight_layout()
        plt.savefig(path_save)
        plt.close()



