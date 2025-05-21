import numpy as np

class Platform():
    def __init__(self, speed: float=0.9, step_size: float=1.0/3600.0, random_seed: int=999999, location_x: float=0.0, location_y: float=0.0, heading: float=3.0*np.pi/4.0):
        self.timestep_size = step_size
        self.mach = speed
        self.seed = self.set_seed(random_seed)
        self._position = (location_x, location_y, heading) #(100.0, 0.0, 3.0*np.pi/4.0)

    def set_seed(self, seed_val) -> int:
        np.random.seed(int(seed_val))
        return seed_val

    def circular_pi(self, value: float) -> float:
        if value > 2*np.pi:
            value = np.remainder(value, 2*np.pi)
        return value
    
    def convert_ft_to_km(self, feet: float) -> float:
        return 0.0003048 * feet
    
    def timestep_update(self, current_time: float) -> bool:
        '''
            Implement on each platform type that inherits from this class
        '''
        pass