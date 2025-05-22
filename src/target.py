import numpy as np
import pandas as pd
from sim_platform import Platform

class TargetPlat(Platform):
    def __init__(self, speed: float=0.9, step_size: float=1.0/3600.0, random_seed: int=999999, location_x: float=0.0, location_y: float=0.0, heading: float=3.0*np.pi/4.0, id: int=0):
        super().__init__(speed=speed, step_size=step_size, random_seed=random_seed, location_x=location_x, location_y=location_y, heading=heading)
        self.speed_kmph = speed * 1225.04   # mach times conversion to kmph
        self.heading = heading
        self._position = (location_x, location_y, heading)
        self.points_log = {'x': [], 'y': [], 'heading':[]}
        self.id = id
        self.tag = ''

    def timestep_update(self, current_time):
        # With current speed in x and y directions and heading, calculate the next position at next time step
        pos = self._position
        new_x = pos[0] + self.timestep_size * self.speed_kmph * np.cos(self.heading)
        new_y = pos[1] + self.timestep_size * self.speed_kmph * np.sin(self.heading)
        self._position = (new_x, new_y, self.heading)
        self.points_log['x'].append(new_x)
        self.points_log['y'].append(new_y)
        self.points_log['heading'].append(self.heading)
        # print(current_time, (new_x, new_y, self.heading))
        return False, np.nan, -1.0
    
    def output_log_points(self):
        pd.DataFrame(self.points_log).to_csv(f'points_log_target_{self.id}_{self.tag}.csv')