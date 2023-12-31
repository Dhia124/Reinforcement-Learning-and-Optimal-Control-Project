import os
import sys
import traci
import numpy as np


class SumoEnvironment:
    def __init__(self, sumo_cfg_file):
        """
        Initialize the Sumo Environment.
        :param sumo_cfg_file: Path to the SUMO configuration file.
        """
        self.sumo_cfg_file = sumo_cfg_file

        # Highway lanes from your network
        self.highway_lanes = [ "E3_0", "E4_0", "E5_0", "E5_1", "E5_2", "E6_0", "E6_1",
                              "E6_2"]

        # Ramp lane from your network (if any)
        # "E3_0" is the ramp lane based on our network structure; replace with the actual ramp lane ID if different
        self.ramp_lane = "E3_0"

        # Traffic light ID at the ramp
        self.traffic_light_id = "J8"  # Based on our network,  J6 is the traffic light ID

        # Ensure SUMO_HOME is set
        self._setup_sumo_home()
        self.state_size = self.define_state_size()
        self.action_size = self.define_action_size()

    def _setup_sumo_home(self):
        if "SUMO_HOME" not in os.environ:
            sys.exit("Please declare the environment variable 'SUMO_HOME'")
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)

    def define_state_size(self):
        # Example: If the state includes vehicle count and average speed for each of 3 lanes
        return 3 * 2  # number of lanes times measurements per lane

    def define_action_size(self):
        # Example: If you have two possible actions for the traffic light (green and red)
        return 2

    def reset(self):
            # Start a new SUMO simulation
            traci.start([traci.constants.CMD_LOAD, os.path.abspath(self.sumo_cfg_file)])
            return self.get_state()

    def step(self, action):
            # Apply action and advance the simulation
            self.apply_action(action)
            traci.simulationStep()
            next_state = self.get_state()
            reward = self.calculate_reward()
            done = traci.simulation.getMinExpectedNumber() <= 0
            return next_state, reward, done

    def get_state(self):
            # Retrieve real-time data about lanes
            state = []
            for lane_id in self.highway_lanes:
                lane_vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                lane_average_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                state.extend([lane_vehicle_count, lane_average_speed])

            ramp_vehicle_count = traci.lane.getLastStepVehicleNumber(self.ramp_lane)
            ramp_average_speed = traci.lane.getLastStepMeanSpeed(self.ramp_lane)
            state.extend([ramp_vehicle_count, ramp_average_speed])

            return np.array(state)

    def calculate_reward(self):
            # Example reward function based on minimizing total waiting time
            total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in self.highway_lanes)
            reward = -total_waiting_time
            return reward

    def apply_action(self, action):
            # Control the traffic light based on the action
            if action == 1:  # Assuming 1 is green and 0 is red
                traci.trafficlight.setPhase(self.traffic_light_id, 0)  # Green phase
            else:
                traci.trafficlight.setPhase(self.traffic_light_id, 2)  # Red phase

    def close(self):
            traci.close()

    # Example usage
if __name__ == "__main__":
        sumo_cfg_file = "path/to/your/haya.sumocfg"
        env = SumoEnvironment(sumo_cfg_file)
        state = env.reset()
        done = False

        while not done:
            action = np.random.choice([0, 1])  # Example: Random action
            next_state, reward, done = env.step(action)

        env.close()
