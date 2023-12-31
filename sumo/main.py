import os
import sys
import traci
from SumoEnvironment import SumoEnvironment
from dqn_agent import DQNAgent

# Make sure the SUMO_HOME environment variable is set
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Initialize the environment and agent
env = SumoEnvironment(r'1.sumocfg')
agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
average_wait_time = 0

# Function to run the simulation
def run_simulation():
    traci.start(['sumo-gui', '-c', env.sumo_cfg_file])
    total_wait_time = 0
    total_vehicles_passed = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        state = env.get_state()
        action = agent.choose_action(state)
        env.apply_action(action)
        reward, wait_time, vehicles_passed = env.step(action)
        total_wait_time += wait_time
        total_vehicles_passed += vehicles_passed

        # Print the current state, action, and reward
        print(f"State: {state}, Action: {action}, Reward: {reward}")

    global average_wait_time
    average_wait_time = total_wait_time / total_vehicles_passed if total_vehicles_passed else 0
    
    traci.close()
    agent.train()
    print("bbbbbb")
    print("aaaaaaaaaaa")
# Main execution
if __name__ == '__main__':
    try:
        run_simulation()
        print("average_wait_time", average_wait_time)
        
        
    except traci.exceptions.FatalTraCIError as e:
        print(f"Error during simulation: {e}")
