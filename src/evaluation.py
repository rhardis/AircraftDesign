import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from pyDOE3 import fullfact as ff

from aircraft import Aircraft

def run_sim(args):
    # unpack the args tuple
    end_time, timestep, sensor, speed, altitude, probability_detect, tgt_location = args

    craft = Aircraft(sensor, speed, altitude, probability_detect, tgt_location)

    current_time = 0.0
    done = False
    while (not done) and (current_time <= end_time):
        # if np.floor(current_time*7200) % 360 == 0:
        #     pos = craft.get_position()[:2]
        #     print(pos)
        # if np.floor(current_time*7200) % 3600 == 0:
        #     print(current_time)
        done, found_time, cost = craft.timestep_update(current_time)
        # if done:
        #     print('done from aircraft condition')
        current_time += timestep

    return pd.DataFrame(data=[[end_time, timestep, sensor, speed, altitude, probability_detect, tgt_location, found_time, cost]],
                        columns=['end_time', 'time_step', 'sensor', 'speed', 'altitude', 'Pdetect', 'target_location', 'Time to Detect', 'Cost ($M)'])

def generate_runs(doe_type, sensor_types, machs, altitudes, end_time, time_step, probability_detect, num_replicates):
    runs = []
    if doe_type == 'ff':
        base_matrix = ff([len(sensor_types),
                          len(machs),
                          len(altitudes)])
    else:
        pass
    return runs

def execute_runs(end_time, time_step, probability_detect, num_replicates=1) -> pd.DataFrame:
    # define the run matrix input factors:
    sensor_types = ['a', 'b', 'c']
    machs = np.linspace(0.4, 0.9, num=5)
    altitudes = np.linspace(5000, 25000, num=5)

    jobs = generate_runs('ff', sensor_types, machs, altitudes, end_time, time_step, probability_detect, num_replicates)

    r = process_map(run_sim, jobs, max_workers=2)
    # run_sim(end_time, time_step, sensor, speed, altitude, probability_detect, tgt_location)


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