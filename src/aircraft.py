# imports
import numpy as np
import pandas as pd

class StaticInfo():
    def __init__(self):
        self.sensor_info: dict = {'a': {'fov': 15, 'cost': 0.05}, 'b': {'fov': 30, 'cost': 1.0}, 'c': {'fov': 60, 'cost': 10.0}}
        self.speed_of_sound_at_altitudes: pd.DataFrame = pd.DataFrame(data=[1204, 1182, 1160, 1138, 1115], index=[5000, 10000, 15000, 20000, 25000], columns=['speed_kmps'])

class Aircraft():
    def __init__(self, sensor_type: str='a', speed: float=0.9, altitude: float=25000, probability_detect: float=0.5, tgt_location_x: float=0.0, tgt_location_y: float=0.0, az_cue_angle: float=-90.0, step_size: float=1.0/7200.0, random_seed: int=999999) -> None:
        self.info = StaticInfo()
        self.timestep_size = step_size
        self.mach = speed
        self.air_speed_kmph = self.convert_mach_to_airspeed(speed, altitude, self.info.speed_of_sound_at_altitudes)
        self.altitude_ft = altitude
        self.altitude_km = self.convert_ft_to_km(altitude)        
        self.sensor_type = sensor_type
        self.sensor_fov = self.info.sensor_info[sensor_type]['fov']
        self.sensor_projected_width = 2 * self.altitude_km * np.tan(0.5 * self.sensor_fov * np.pi / 180)
        self.sensor_cost = self.info.sensor_info[sensor_type]['cost']
        self.az_cue_angle = az_cue_angle
        self.one_way_transit_time_hrs = (100 / self.air_speed_kmph)
        self.max_on_station_endurance = self.calc_max_on_station_endurance()
        self._remaining_endurance = self.max_on_station_endurance
        self.p_detect = probability_detect
        self.target_location = (tgt_location_x, tgt_location_y)
        self._position = (100.0, 0.0, 3.0*np.pi/4.0)
        self.route = self.make_initial_route()
        self.current_waypoint = 0
        self.seed = self.set_seed(random_seed)

    def set_seed(self, seed_val) -> int:
        np.random.seed(seed_val)
        return seed_val

    def circular_pi(self, value: float) -> float:
        if value > 2*np.pi:
            value = np.remainder(value, 2*np.pi)
        return value

    def make_initial_route(self) -> list[tuple[float]]:
        '''
            Makes a route of positions to go to in the form of (x, y, theta) where x points right, y up, and theta is angle counterclockwise from x axis
        '''
        center_x = 50
        center_y = 50
        pos = (center_x, center_y, np.pi)
        route = [pos]

        D_val = self.sensor_projected_width

        done = False
        counter = 0
        while not done:
            if counter <= 1:
                D_prior = D_val
            else:
                D_prior = D_val * np.floor((counter + 3) / 2)

            next_x = pos[0] + np.cos(pos[2]) * D_prior
            next_y = pos[1] + np.sin(pos[2]) * D_prior
            next_theta = self.circular_pi(pos[2] + np.pi/2)

            pos = (next_x, next_y, next_theta)
            route.append(pos)
            counter += 1

            if np.abs((center_x - pos[0])) > center_x and np.abs((center_y - pos[1])) > center_y:
                done = True            
        
        return route

    def convert_ft_to_km(self, feet: float) -> float:
        return 0.0003048 * feet

    def set_remaining_endurance(self, time: float) -> None:
        self._remaining_endurance = self.max_on_station_endurance - time

    def get_remaining_endurance(self) -> bool:
        return self._remaining_endurance

    def timestep_update(self, current_time: float) -> bool:
        '''
            Calculate a timestep update and end sim if target found or sim time has ended
        '''
        # check if found target
        target_found_time = np.nan
        tgt_found: bool = self.check_point_in_fov(self.target_location)

        # if found target, update the results information and end the sim
        if tgt_found:
            # print(f'target found when craft was at {self.get_position()[:2]} after {current_time:.2f} hours.')
            target_found_time = (current_time + self.one_way_transit_time_hrs).iloc[0]  # convert to numpy float 64 from pandas series with 1 element
            return True, target_found_time, self.calc_design_cost()

        # if BINGO status, end the sim
        elif current_time >= self.max_on_station_endurance:
            # print('ran out of fuel')
            return True, target_found_time, self.calc_design_cost()
        
        # the target is not found and there is remaining endurance.  Calculate the next position
        else:
            # update the position
            self.set_position(self.calculate_next_position())

            # 
        return False, target_found_time, -1.0

    def get_position(self) -> tuple[float]:
        return self._position

    def set_position(self, new_pos: tuple[float]) -> None:
        self._position = new_pos

    def calculate_next_position(self) -> tuple[float]:
        goal_waypoint = self.route[self.current_waypoint]
        upcoming_waypoint_heading = goal_waypoint[2]
        current_position = self.get_position()
        current_xy = current_position[:2]
        heading = current_position[2]

        # check if reached next waypoint.  If so, update to next waypoint
        dist_from_waypoint = np.linalg.norm([goal_waypoint[0] - current_xy[0], goal_waypoint[1] - current_xy[1]])
        if dist_from_waypoint <= 0.5:
            heading = upcoming_waypoint_heading
            current_position = goal_waypoint
            
            # set the next waypoint
            self.current_waypoint += 1

        # With current speed in x and y directions and heading, calculate the next position at next time step
        new_x = current_xy[0] + self.timestep_size * self.air_speed_kmph * np.cos(heading)
        new_y = current_xy[1] + self.timestep_size * self.air_speed_kmph * np.sin(heading)
        return (new_x, new_y, heading)

    def convert_mach_to_airspeed(self, mach_number: float, altitude_ft: float, mach_speed_table: pd.DataFrame) -> float:
        '''
            Converts a mach number and altitude to an airspeed in km/s
        '''
        for upper_alt in mach_speed_table.index[1:]:
            lower_alt = upper_alt - 5000
            upper_speed = mach_speed_table.loc[upper_alt]
            lower_speed = mach_speed_table.loc[lower_alt]
            if altitude_ft <= upper_alt:
                break
        

        return mach_number * (((altitude_ft - lower_alt) / (upper_alt - lower_alt) * (upper_speed - lower_speed)) + lower_speed)

    def calc_max_on_station_endurance(self) -> float:
        unavailable_transit_time = self.one_way_transit_time_hrs * 2
        total_endurance = (-18.75 * (self.mach**2)) + (8.0893 * self.mach) + (0.01 * (self.altitude_ft / 1000)**2) + (0.05 * self.altitude_ft / 1000) + (9.2105)
        on_station_endurance = total_endurance - unavailable_transit_time
        return on_station_endurance.iloc[0]
    
    def calc_design_cost(self) -> float:
        '''
            Uses the aircraft's cost model and sensor cost model to create total cost
        '''
        aircraft_cost = (50 * self.mach**2) - (35 * self.mach) + (0.03 * (self.altitude_ft/1000)**2) - (0.2 * (self.altitude_ft/1000)) + 11
        total_cost = aircraft_cost + self.sensor_cost
        return total_cost

    def check_point_in_fov(self, point: tuple[float]) -> bool:
        '''
        
        '''
        # locations:
        tgt_x = self.target_location[0]
        tgt_y = self.target_location[1]
        own_x = self.get_position()[0]
        own_y = self.get_position()[1]        

        # translate relative to sensor platform
        relative_x = tgt_x - own_x
        relative_y = tgt_y - own_y

        # rotate the coordinates from WCS to entity's coordinate system x,y --> u,v
        psi = self.get_position()[2] - np.arctan2(relative_y , relative_x)
        psi_deg = psi * 180 / np.pi
        norm = np.sqrt(relative_x**2 + relative_y**2)
        u = norm * np.cos(psi)
        v = norm * np.sin(psi)

        # check the point's u,v coordinates are both within altitude * tan(1/2 FOV)
        if np.all([(np.abs(u) <= self.sensor_projected_width) , (np.abs(v) <= self.sensor_projected_width)]):
            random_draw_successful_detect = np.random.binomial(1, self.p_detect)
            if(random_draw_successful_detect):
                # print('successful detect')
                return True
            else:
                # print('unsuccessful detect in range')
                pass

        # simplified method:

        # dist = np.linalg.norm([tgt_x - own_x, tgt_y - own_y])
        # if dist <= self.sensor_projected_width:
        #     return True
        return False
    

