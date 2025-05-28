import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_aircraft_route(air_df, tgt_dfs, plot_tgts, plot_route, plot_fov):
    if plot_route:
        axes = air_df.plot.line(x='x', y='y')

    else:
        fig, axes = plt.subplots()
    
    air_start_point = air_df.iloc[0, 1:3]    
    air_end_point = air_df.iloc[-1, 1:3]    
    prj_w = air_df['projection_width'][0]
    square = patches.Rectangle((air_end_point['x'] - prj_w/2, air_end_point['y'] - prj_w/2), prj_w, prj_w, edgecolor='purple', facecolor='none')
    axes.add_patch(square)


    for tgt_df in tgt_dfs:
        plt.plot(tgt_df['x'], tgt_df['y'], color='red', alpha=0.5)
    
        tgt_end_point = tgt_df.iloc[-1, 1:3]
        x = [air_end_point.loc['x'], tgt_end_point.loc['x']]
        y = [air_end_point.loc['y'], tgt_end_point.loc['y']]
        plt.plot(x, y, color='green')
        plt.scatter(tgt_end_point['x'], tgt_end_point['y'], marker='v', color='r')
    plt.scatter(air_end_point['x'], air_end_point['y'], marker='^', color='b')
    plt.scatter(air_start_point['x'], air_start_point['y'], marker='o', color='black')
    plt.text(100.0, 0.0, 'Start Point')
    axes.legend(labels=['search path', 'EOIR Sea Level FOV', 'tgt movement path(s)'])

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title('Aircraft Spiral Search Pattern\nand Target Movement in Search Area')
    plt.savefig('search path.png')
    plt.show()

num_tgts = {0.0: 1, 1.0: 1, 2.0: 10}

if __name__ == '__main__':
    altitude = 5000.0
    speed = 0.4
    sensor = 'c'
    mission = 0.0
    tag = f'{float(altitude)}_{float(speed)}_{sensor}_{float(mission)}'
    target_dfs = []
    for i in range(num_tgts[mission]):
        target_file = f'points_log_target_{i}_{tag}.csv'
        tgt_df = pd.read_csv(target_file)
        target_dfs.append(tgt_df)

    aircraft_file = f'points_log_aircraft_{tag}.csv'
    air_df = pd.read_csv(aircraft_file)

    plot_tgts = True
    plot_route = True
    plot_fov = True
    plot_aircraft_route(air_df, target_dfs, plot_tgts, plot_route, plot_fov)