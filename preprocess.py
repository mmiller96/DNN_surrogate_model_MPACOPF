import pandas as pd
import pandapower as pp
import numpy as np
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    entsoe_load_path = os.path.join(script_dir, "data", "entsoe_load_consumption_countries.csv")
    weather_path = os.path.join(script_dir, "data", "weather_Löningen.csv")
    price_path = os.path.join(script_dir, "data", "Gross_handelspreise.csv")
    final_path = os.path.join(script_dir, "data", "ieee33.csv")

    entsoe_load = pd.read_csv(entsoe_load_path, index_col=0)
    weather = pd.read_csv(weather_path, index_col=0)
    prices = pd.read_csv(price_path, sep=';')
    
    prices_germany = prices['Deutschland/Luxemburg [€/MWh] Originalauflösungen']
    load_germany = entsoe_load['DE_tennet_load_actual_entsoe_transparency'].iloc[1:]
    load_germany.index = pd.to_datetime(load_germany.index)
    weather.index = pd.to_datetime(weather.index)
    prices_index = pd.to_datetime(prices['Datum von'], format='%d.%m.%Y %H:%M')

    load_germany_2019 = load_germany[(load_germany.index >= '2019-01-01') & (load_germany.index < '2020-01-01')]
    weather_2019 = weather[(weather.index >= '2019-01-01') & (weather.index < '2020-01-01')]
    prices_germany_2023 = prices_germany[(prices_index >= '2023-01-01') & (prices_index < '2024-01-01')]
    prices_germany_2023.index = weather_2019.index  # use indices of 2019 (weather data of 2023 were not available)
    prices_germany_2023[:] = np.char.replace(prices_germany_2023.values.astype(str), ',', '.').astype(float)
    net = pp.networks.case33bw()

    load_data = net.load.copy()
    data_2019 = load_germany_2019/load_germany_2019.max()
    num_buses = net.bus.shape[0]

    # Initialize DataFrames for P and Q with zeros
    P_columns = ['P_' + str(i) for i in range(num_buses)]
    Q_columns = ['Q_' + str(i) for i in range(num_buses)]

    P_df = pd.DataFrame(0, index=weather_2019.index, columns=P_columns)
    Q_df = pd.DataFrame(0, index=weather_2019.index, columns=Q_columns)

    scaling_factor = 1.0
    for idx, load in load_data.iterrows():
        bus_idx = load['bus']
        P_peak = load['p_mw']* scaling_factor   # Peak active power at this bus
        Q_peak = load['q_mvar']* scaling_factor # Peak reactive power at this bus
        P_df['P_' + str(bus_idx)] = data_2019.values * P_peak
        Q_df['Q_' + str(bus_idx)] = data_2019.values * Q_peak
    final_df = pd.concat([P_df, Q_df, weather_2019, prices_germany_2023], axis=1)
    final_df.to_csv(final_path)

