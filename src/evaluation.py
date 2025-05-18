import numpy as np
import pandas as pd

from aircraft import Aircraft

def run_sim(end_time, timestep, sensor, speed, altitude, probability_detect, tgt_location):
    craft = Aircraft(sensor, speed, altitude, probability_detect, tgt_location)

    current_time = 0.0
    done = False
    while (not done) and (current_time <= end_time):
        # if np.floor(current_time*7200) % 360 == 0:
        #     pos = craft.get_position()[:2]
        #     print(pos)
        if np.floor(current_time*7200) % 3600 == 0:
            print(current_time)
        done, found_time, cost = craft.timestep_update(current_time)
        if done:
            print('done from aircraft condition')
        current_time += timestep

    return pd.DataFrame(data=[[end_time, timestep, sensor, speed, altitude, probability_detect, tgt_location, found_time, cost]],
                        columns=['end_time', 'time_step', 'sensor', 'speed', 'altitude', 'Pdetect', 'target_location', 'Time to Detect', 'Cost ($M)'])

if __name__ == "__main__":
    sensor = 'c'
    speed = 0.7
    altitude = 25000
    probability_detect = 0.1
    tgt_location = (100.0, 50.0)
    
    end_time = 18   # hours
    time_step = 1.0 / 7200.0    # 0.5 seconds
    result = run_sim(end_time, time_step, sensor, speed, altitude, probability_detect, tgt_location)
    print(result)