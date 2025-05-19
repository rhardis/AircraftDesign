import pandas as pd
import matplotlib.pyplot as plt
from os import path
from paretoset import paretoset
from sklearn.preprocessing import LabelEncoder

def print_summaries_by_sensor(df):
    for sensor in ['a', 'b', 'c']:
        df_sample = df[df['sensor'] == sensor]
        df_sample = df_sample[['Time to Detect', 'Cost ($M)']]
        print(sensor)
        print(df_sample.describe())
        print('\n\n')

def plot_pareto_points(df, x, y):
    df = df[[x, y]]

    mask = paretoset(df, sense=["min", "min"])
    paretoset_points = df[mask]
    df['optimality'] = 0
    df.iloc[paretoset_points.index.values, -1] = 1

    df.plot(kind='scatter', x=x, y=y, c='optimality', cmap='viridis')
    plt.title(f'Pareto Optimal {y} - {x} Tradeoff')
    plt.xlabel('Time to Detect')
    plt.ylabel('Cost ($M)')
    plt.show()

def aggregate_over_treatment(df) -> pd.DataFrame:
    '''
    '''

    df = df.groupby(by='treatment').mean()
    df.index = df.index.astype(int)
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
    load_file = 'simulation_results_72.0_treatments_3_reps.csv'
    df = pd.read_csv(path.join(load_file))
    df = encode_categorical(df, ['sensor'])
    df = aggregate_over_treatment(df)
    print_summaries_by_sensor(df)
    plot_pareto_points(df, 'Time to Detect', 'Cost ($M)')