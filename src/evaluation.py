import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from pyDOE3 import fullfact as ff

from aircraft import Aircraft

def run_sim(args):
    # unpack the args tuple
    end_time, timestep, sensor, speed, altitude, probability_detect, tgt_x, tgt_y, random_seed = args

    craft = Aircraft(sensor, speed, altitude, probability_detect, tgt_x, tgt_y, random_seed)

    current_time = 0.0
    done = False
    while (not done) and (current_time <= end_time):
        # if np.floor(current_time*7200) % 360 == 0:
        #     pos = craft.get_position()[:2]
        #     print(pos)
        if np.floor(current_time*7200) % 1800 == 0:
            print(f'{current_time:.2f}')
        done, found_time, cost = craft.timestep_update(current_time)
        # if done:
        #     print('done from aircraft condition')
        current_time += timestep

    return pd.DataFrame(data=[[end_time, timestep, sensor, speed, altitude, probability_detect, tgt_x, tgt_y, found_time, cost]],
                        columns=['end_time', 'time_step', 'sensor', 'speed', 'altitude', 'Pdetect', 'target_x', 'target_y', 'Time to Detect', 'Cost ($M)'])

def generate_runs(doe_type, sensor_types, machs, altitudes, end_time, time_step, probability_detect, num_replicates):
    runs = []
    if doe_type == 'ff':
        
        base_matrix = ff([len(sensor_types),
                          len(machs),
                          len(altitudes)])
        base_matrix = pd.DataFrame(base_matrix)
        for row in range(base_matrix.shape[0]):
            base_matrix.iloc[row, 0] = sensor_types[int(base_matrix.iloc[row, 0])]
            base_matrix.iloc[row, 1] = machs[int(base_matrix.iloc[row, 1])]
            base_matrix.iloc[row, 2] = altitudes[int(base_matrix.iloc[row, 2])]

        ones_mat = pd.DataFrame(np.ones(shape=(base_matrix.shape[0], 9))) # 9 total columns to send into the run_sim function
        ones_mat.iloc[:, 2] = ones_mat.iloc[:, 2].astype(str)
        ones_mat.iloc[:, 2:5] = base_matrix
        ones_mat.iloc[:, 0] = end_time
        ones_mat.iloc[:, 1] = time_step
        ones_mat.iloc[:, 5] = probability_detect
        ones_mat.iloc[:, 6] = tgt_location[0]
        ones_mat.iloc[:, 7] = tgt_location[1]
        for rand_seed in range(1, num_replicates+1):
            ones_mat.iloc[:, 8] = rand_seed
            for idx in range(ones_mat.shape[0]):
                runs.append(ones_mat.iloc[idx, :].to_list())
    else:
        pass
    return runs

def execute_runs(end_time, time_step, probability_detect, num_replicates=1) -> pd.DataFrame:
    # define the run matrix input factors:
    sensor_types = ['a', 'b', 'c']
    machs = np.linspace(0.4, 0.9, num=5)
    altitudes = np.linspace(5000, 25000, num=5)

    jobs = generate_runs('ff', sensor_types, machs, altitudes, end_time, time_step, probability_detect, num_replicates)

    result = run_sim(jobs[0])
    
    r = process_map(run_sim, jobs, max_workers=2)

    results_df = pd.DataFrame(columns=['end_time', 'time_step', 'sensor', 'speed', 'altitude', 'Pdetect', 'target_x', 'target_y', 'Time to Detect', 'Cost ($M)'])
    for result in r:
        results_df = pd.concat([results_df, result])
    results_df.to_csv(f'simulation_results_{len(jobs)/num_replicates}_treatments_{num_replicates}_reps.csv')
    return results_df

if __name__ == "__main__":
    num_reps = 2

    sensor = 'c'
    speed = 0.7
    altitude = 25000
    probability_detect = 0.1
    tgt_location = (100.0, 50.0)
    
    end_time = 18   # hours
    time_step = 1.0 / 7200.0    # 0.5 seconds

    result: pd.DataFrame = execute_runs(end_time=end_time, time_step=time_step, probability_detect=probability_detect, num_replicates=num_reps)
    print(result)