import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from pyDOE3 import fullfact as ff

from aircraft import Aircraft
from target import TargetPlat

def run_sim(args):
    # unpack the args tuple
    end_time, timestep, sensor, speed, altitude, mission_type, probability_detect, tgt_x, tgt_y, random_seed, treatment = args

    if mission_type == 0.0:
        # SAR
        print('SAR')
        tgt_heading = np.random.random()*np.pi*2    # an angle 0 to 360
        tgt_speed = 0.1 / 1225.04   # speed in mach as sea level.  Moves at 0.1 km/h
        targets = {0: {'tgt_object': TargetPlat(speed=tgt_speed, step_size=time_step, random_seed=random_seed, location_x=tgt_x, location_y=tgt_y, heading=tgt_heading), 
                       'found': {'time': np.nan, 'value': False}}}
    elif mission_type == 1.0:
        # interdiction
        print('Interdiction')
        tgt_heading = np.random.random()*np.pi*2    # an angle 0 to 360
        tgt_speed = 10.0 / 1225.04   # speed in mach as sea level.  Moves at 5 km/h
        targets = {0: {'tgt_object': TargetPlat(speed=tgt_speed, step_size=time_step, random_seed=random_seed, location_x=tgt_x, location_y=tgt_y, heading=tgt_heading), 
                       'found': {'time': np.nan, 'value': False}}}        
    else:
        # counter-swarm
        print('C-USV')
        # adjust the usvs to start within x and y [40, 60] so they don't exit the area too early
        tgt_x = ((tgt_x / 100) * (60 - 40)) + 40
        tgt_y = ((tgt_y / 100) * (60 - 40)) + 40
        tgt_heading = np.random.random()*np.pi*2    # an angle 0 to 360
        tgt_speed = 5.0 / 1225.04   # speed in mach as sea level.  Moves at 5 km/h

        targets = {}      
        for i in range(10):
            tgt_x += np.random.uniform(low=-1.0, high=1.0) * 5
            tgt_y += np.random.uniform(low=-1.0, high=1.0) * 5
            targets[i] = {'tgt_object': TargetPlat(speed=tgt_speed, step_size=time_step, random_seed=random_seed, location_x=tgt_x, location_y=tgt_y, heading=tgt_heading), 
                       'found': {'time': np.nan, 'value': False}}

    craft = Aircraft(sensor, speed, altitude, probability_detect, 100.0, 0.0, random_seed, step_size=timestep, targets=targets)

    current_time = 0.0
    done = False
    while (not done) and (current_time <= end_time):
        # if np.floor(current_time*7200) % 360 == 0:
        #     pos = craft.get_position()[:2]
        #     print(pos)
        # if np.floor(current_time*7200) % 1800 == 0:
        #     print(f'{current_time:.2f}')
        for target_id, info in targets.items():
            info['tgt_object'].timestep_update(current_time)
        done, found_time, cost = craft.timestep_update(current_time)
        if done:
            print('done from aircraft condition')
        current_time += timestep

    return pd.DataFrame(data=[[treatment, random_seed, end_time, timestep, sensor, speed, altitude, probability_detect, tgt_x, tgt_y, found_time, cost, mission_type]],
                        columns=['treatment', 'replicate', 'end_time', 'time_step', 'sensor', 'speed', 'altitude', 'Pdetect', 'target_x', 'target_y', 'Time to Detect', 'Cost ($M)', 'mission_type'])

def generate_runs(doe_type, sensor_types, machs, altitudes, missions, end_time, time_step, probability_detect, num_replicates):
    runs = []
    if doe_type == 'ff':
        
        base_matrix = ff([len(sensor_types),
                          len(machs),
                          len(altitudes),
                          len(missions)])
        base_matrix = pd.DataFrame(base_matrix)
        for row in range(base_matrix.shape[0]):
            base_matrix.iloc[row, 0] = sensor_types[int(base_matrix.iloc[row, 0])]
            base_matrix.iloc[row, 1] = machs[int(base_matrix.iloc[row, 1])]
            base_matrix.iloc[row, 2] = altitudes[int(base_matrix.iloc[row, 2])]
            base_matrix.iloc[row, 3] = missions[int(base_matrix.iloc[row, 3])]

        ones_mat = pd.DataFrame(np.ones(shape=(base_matrix.shape[0], 11))) # 10 total columns to send into the run_sim function
        ones_mat.iloc[:, 2] = ones_mat.iloc[:, 2].astype(str)
        ones_mat.iloc[:, 2:6] = base_matrix
        ones_mat.iloc[:, 0] = end_time
        ones_mat.iloc[:, 1] = time_step
        ones_mat.iloc[:, 6] = probability_detect
        ones_mat.iloc[:, 10] = ones_mat.index   # set the treatment number
        ones_mat.iloc[:, 9] = ones_mat.iloc[:, 9].astype(int)

        # for each replicate:
        for rand_seed in range(1, num_replicates+1):
            # set the random values for x y tgt position and random seed
            np.random.seed(rand_seed)
            ones_mat.iloc[:, 7] = np.random.random(size=base_matrix.shape[0]) * 100
            ones_mat.iloc[:, 8] = np.random.random(size=base_matrix.shape[0]) * 100            
            ones_mat.iloc[:, 9] = int(rand_seed)
            for idx in range(ones_mat.shape[0]):
                runs.append(ones_mat.iloc[idx, :].to_list())
    else:
        pass
    return runs

def execute_runs(end_time, time_step, probability_detect, num_replicates=1) -> pd.DataFrame:
    # define the run matrix input factors:
    sensor_types = ['a', 'b', 'c']
    machs = np.linspace(0.4, 0.9, num=4)
    altitudes = np.linspace(5000, 25000, num=6)
    missions = [0]

    jobs = generate_runs('ff', sensor_types, machs, altitudes, missions, end_time, time_step, probability_detect, num_replicates)

    run_sim(jobs[0])

    print(f'running {len(jobs)} jobs on {mp.cpu_count()-1} cores.')
    r = process_map(run_sim, jobs, max_workers=mp.cpu_count()-1, chunksize=10)

    results_df = pd.DataFrame(columns=['end_time', 'time_step', 'sensor', 'speed', 'altitude', 'Pdetect', 'target_x', 'target_y', 'Time to Detect', 'Cost ($M)'])
    for result in r:
        results_df = pd.concat([results_df, result])
    results_df.to_csv(f'simulation_results_{len(jobs)/num_replicates}_treatments_{num_replicates}_reps.csv', index=False)
    return results_df

if __name__ == "__main__":
    num_reps = 2
    probability_detect = 0.5
    end_time = 18   # hours
    time_step = 1.0 / 3600.0    # 0.5 seconds

    result: pd.DataFrame = execute_runs(end_time=end_time, time_step=time_step, probability_detect=probability_detect, num_replicates=num_reps)
    print(result)