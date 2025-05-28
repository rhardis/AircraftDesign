import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from paretoset import paretoset
from sklearn.preprocessing import LabelEncoder

def get_sensor_type(label: float) -> str:
    mapping = {1.0: 'a', 2.0: 'b', 3.0: 'c'}
    return mapping[label]

def get_mission_type(label: float) -> str:
    mapping = {1.0: 'Maritime Smuggling Interdiction USCG', 2.0: 'Counter-USV DILR USN', 0.0: 'Search and Rescue USCG'}
    return mapping[label]

def print_summaries_by_sensor(df):
    for sensor in [1.0, 2.0, 3.0]:
        df_sample = df[df['sensor'] == sensor]
        df_sample = df_sample[['Time to Detect', 'Cost_($M)']]
        print(sensor)
        print(df_sample.describe())
        print('\n\n')

def plot_score_cost_tradeoff(df, x, y):
    sense = ["max", "min"]
    text_offsets = (30, -50)

    df.reset_index(inplace=True, drop=True)
    info_df = df.copy(deep=True)
    info_df['sensor'] /= 3
    info_df['speed'] /=3
    info_df['altitude'] /= 3
    info_df['Cost_($M)'] /= 3
    df = df[[x, y]]
    df['Cost_($M)'] /= 3
    mask = paretoset(df, sense=sense)
    paretoset_points = df[mask]

    df['optimality'] = (np.sqrt((df[x])**2 + (df[y] - 45)**2) / np.sqrt(df[x].max()**2 + df[y].max()**2)) * 0.7
    df.iloc[paretoset_points.index.values, -1] = 1

    df.plot(kind='scatter', x=y, y=x, c='optimality', cmap='Greys', figsize=(10.5, 5.5))
    optimal_df = df[df['optimality'] == 1].sort_values(by=x)
    plt.plot(optimal_df[y], optimal_df[x], color='black')
    plt.scatter(optimal_df[y], optimal_df[x], color='green', alpha=0.3, s=100)

    for point in optimal_df.index:
        x_loc = optimal_df.loc[point, x]
        y_loc = optimal_df.loc[point, y]
        err = info_df.loc[point, 'total_std']
        speed_data = info_df.loc[point, 'speed']
        alt_data = info_df.loc[point, 'altitude']
        sensor_data = get_sensor_type(info_df.loc[point, 'sensor'])
        text = f'sensor: {sensor_data}\nspeed: {speed_data:.1f}\naltitude: {alt_data:.1f}'

        if use_err_bars:
            plt.errorbar(x=y_loc, y=x_loc, yerr=err, color='lightgray', capsize=5)

        if use_labels:
            plt.annotate(text, (y_loc, x_loc), textcoords="offset points", xytext=text_offsets, ha='center',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=5,
                     bbox=dict(boxstyle='square,pad=0.1', edgecolor='black', facecolor='white'))
        
    common_point = info_df[(info_df['altitude'] == 21000) & (info_df['sensor'] == 2.0) & (info_df['speed'] >= 0.5) & (info_df['speed'] < 0.6)]
    common_point.reset_index(inplace=True, drop=True)
    plt.scatter(common_point.loc[0, y], common_point.loc[0, x], marker='o', color='blue', s=100)
    plt.scatter(common_point.loc[0, y], common_point.loc[0, x], marker='*', color='yellow', s=100)
    
    score_function = f'total_score = (\n    10 * Search and Rescue Score [0, 1] +\n    10 * Drug Interdiction Score [0, 1] +\n    80 * Counter USV Score [0, 1]\n)'
    plt.annotate(score_function, (20, 20), bbox=dict(boxstyle='square,pad=0.3', edgecolor='black', facecolor='lightgray'))
        
    title = f'Pareto Optimal {y} - {x} Tradeoff\nfor All Mission Types Weighted Total Score'
    save_title = f'Pareto Optimal {y} - {x} Tradeoff for All Mission Types Weighted Total Score.png'
    plt.title(title)
    plt.xlabel(y)
    plt.ylabel(x)
    plt.savefig(save_title, dpi=300)
    

def plot_pareto_points(df, x, y, cutoff):
    df.reset_index(inplace=True, drop=True)
    df.fillna(20, inplace=True)
    mission_label = df.loc[0, 'mission_type']

    if mission_label == 0:
        sense = ["min", "min"]
        text_offsets = (50, 50)
        df = df[df[x] <= cutoff]
    else:
        sense = ["max", "min"]
        text_offsets = (50, -50)
        df = df[df[x] >= cutoff]

    df.reset_index(inplace=True, drop=True)
    info_df = df.copy(deep=True)
    df = df[[x, y]]

    mask = paretoset(df, sense=sense)
    paretoset_points = df[mask]
    df['optimality'] = (np.sqrt((df[x])**2 + (df[y] - 45)**2) / np.sqrt(df[x].max()**2 + df[y].max()**2)) * 0.7
    df.iloc[paretoset_points.index.values, -1] = 1

    df.plot(kind='scatter', x=y, y=x, c='optimality', cmap='Greys', figsize=(9, 5.5))
    optimal_df = df[df['optimality'] == 1].sort_values(by=x)
    plt.plot(optimal_df[y], optimal_df[x], color='black')
    plt.scatter(optimal_df[y], optimal_df[x], color='green', alpha=0.3, s=100)

    for point in optimal_df.index:
        x_loc = optimal_df.loc[point, x]
        y_loc = optimal_df.loc[point, y]
        err = info_df.loc[point, f'{x}_std']

        if use_err_bars:
            plt.errorbar(y_loc, x_loc, yerr=err, color='lightgray', capsize=5)

        speed_data = info_df.loc[point, 'speed']
        alt_data = info_df.loc[point, 'altitude']
        sensor_data = get_sensor_type(info_df.loc[point, 'sensor'])
        text = f'sensor: {sensor_data}\nspeed: {speed_data:.1f}\naltitude: {alt_data:.1f}'

        if use_labels:
            plt.annotate(text, (y_loc, x_loc), textcoords="offset points", xytext=text_offsets, ha='center',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=5)

    common_point = info_df[(info_df['altitude'] == 21000) & (info_df['sensor'] == 2.0) & (info_df['speed'] >= 0.5) & (info_df['speed'] < 0.6)]
    common_point.reset_index(inplace=True, drop=True)
    plt.scatter(common_point.loc[0, y], common_point.loc[0, x], marker='o', color='blue', s=100)
    plt.scatter(common_point.loc[0, y], common_point.loc[0, x], marker='*', color='yellow', s=100)

    title = f'Pareto Optimal {y} - {x} Tradeoff\nfor Mission Type: "{get_mission_type(mission_label)}"'
    save_title = f'Pareto Optimal {y} - {x} Tradeoff for Mission Type {get_mission_type(mission_label)}.png'
    plt.title(title)
    plt.xlabel(y)
    plt.ylabel(x)
    plt.savefig(save_title, dpi=300)

def aggregate_over_treatment(df) -> pd.DataFrame:
    '''
    '''

    mean_df = df.groupby(by='treatment').mean()
    mean_df.index = mean_df.index.astype(int)

    std_df = df.groupby(by='treatment').std()
    std_df.index = std_df.index.astype(int)

    mean_df['Time_to_Detect_hrs_std'] = std_df['Time_to_Detect_hrs']
    mean_df['Targets_Found_std'] = std_df['Targets_Found']
    return mean_df

def score_df(df):
    scores = [10, 10, 80]
    df.fillna(20, inplace=True)
    df['score_constant'] = [scores[int(mission)] for mission in df['mission_type']]
    df['score_val'] = 0
    df['score_std'] = 0
    counter = 0
    for row in df.itertuples():
        '''
        '''
        if row.mission_type == 0:
            score = (20.0 - row.Time_to_Detect_hrs) / 20.0
            var = (20.0 - row.Time_to_Detect_hrs_std) / 20.0
        elif row.mission_type == 1:
            score = row.Targets_Found / 1.0
            var = row.Targets_Found_std / 1.0
        else:
            score = row.Targets_Found / 10.0
            var = row.Targets_Found_std / 10.0
        
        df.loc[counter, 'score_val'] = score
        df.loc[counter, 'score_std'] += var
        counter += 1

    df['total_score'] = df['score_val'] * df['score_constant']
    df['total_std'] = df['score_std'] * df['score_constant']

    df['tag'] = [f'{sensor}_{speed}_{alt}' for sensor, alt, speed in zip(df['sensor'], df['speed'], df['altitude'])]

    df = df.groupby(by='tag').sum()
    df['total_std'] = np.sqrt(df['total_std'])

    df.sort_values(by='total_score', inplace=True, ascending=False)
    return df

def encode_categorical(df, encode_cols: list[str]) -> pd.DataFrame:
    '''
    '''

    label_encoder = LabelEncoder()
    for col in encode_cols:
        df[col] = label_encoder.fit_transform(df[col])
        if col == 'sensor':
            df[col] += 1
    return df

if __name__ == '__main__':
    use_labels = False
    use_err_bars = True

    load_file = 'simulation_results_216.0_treatments_4_reps.csv'
    df = pd.read_csv(path.join(load_file))
    df = encode_categorical(df, ['sensor'])
    base_df = aggregate_over_treatment(df)

    scored_df = score_df(base_df)
    plot_score_cost_tradeoff(scored_df, 'total_score', 'Cost_($M)')

    # print_summaries_by_sensor(df)
    for option, cutoff in zip([0, 1, 2], [10, 0.5, 5]):
        if option == 0:
            metric = 'Time_to_Detect_hrs'
        else:
            metric = 'Targets_Found'
        plot_df = base_df[base_df['mission_type'] == option]
        plot_pareto_points(plot_df, metric, 'Cost_($M)', cutoff)
