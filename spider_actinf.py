import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sys
sys.path.append(r'E:\phd-proj\dth\pymdp\pymdp')

from pymdp import utils
from pymdp.agent import Agent
from pymdp.maths import softmax

global grid_size

#Spider class
class Spider:  
    def __init__(self, start_position, web_state=0):
        self.position = start_position  # The spider's current position
        self.web_state = web_state  # The spider's web state
    
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        assert isinstance(value, tuple) and len(value) == 2, "Position must be a tuple of length 2"
        self._position = value

    def get_position(self):
        # Return the position as a tuple (x, y)
        return self.position

    def move(self, action, web):
        # Update the spider's position and web state based on the action
        x_action, y_action, web_action = action  # Separate the action into x, y, and web components

        # Calculate new position
        new_position = list(self.position)
        if x_action == 0 and self.position[1] > 0:  # Move left
            new_position[1] -= 1
        elif x_action == 1 and self.position[1] < grid_size - 1:  # Move right
            new_position[1] += 1
        if y_action == 0 and self.position[0] > 0:  # Move up
            new_position[0] -= 1
        elif y_action == 1 and self.position[0] < grid_size - 1:  # Move down
            new_position[0] += 1

        # Check if new position is a valid cell in the web
        if tuple(new_position) in web.cells:
            self.position = tuple(new_position)

        # Update web state
        if web_action == 1:  # Spin web
            self.web_state = 1
        else:  # Stop spinning web
            self.web_state = 0

        # If the spider is spinning a web, add its current position to the web's cells
        if self.web_state == 1:
            web.update_state(self.position, grid_size)

class Web:
    def __init__(self, center):
        self.center = center  # The center of the web
        self.cells = []  # The cells that are part of the web

    def initialize_web_layout(self, grid_size):
        # Build the web in a radial style with 2 levels of depth
        for dx in range(-2, 3):  # dx is the change in x-coordinate
            for dy in range(-2, 3):  # dy is the change in y-coordinate
                # Calculate the distance from the center
                distance = abs(dx) + abs(dy)
                # If the distance is less than or equal to 2, add the cell to the web
                if distance <= 2:
                    x = self.center[0] + dx
                    y = self.center[1] + dy
                    # Check if the cell is within the grid boundaries
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        self.cells.append((x, y))
        # Remove duplicates
        self.cells = list(set(self.cells))

    def update_state(self, new_cell, grid_size):
        # Check if the new cell is within the grid boundaries
        if 0 <= new_cell[0] < grid_size and 0 <= new_cell[1] < grid_size:
            # Add the new cell to the web
            self.cells.append(new_cell)
            # Remove duplicates
            self.cells = list(set(self.cells))

class Environment:
    def __init__(self, grid_size):
        self.grid_size = grid_size  # The size of the grid in the environment
        self.spiders = []  # The spiders in the environment
        self.webs = []  # The webs in the environment

    def add_spider(self, spider):
        # Add a spider to the environment
        self.spiders.append(spider)

    def add_web(self, web):
        # Add a web to the environment
        self.webs.append(web)

    def observe(self):
        # Get the current position of the spider
        spider_position = self.spiders[0].get_position()

        # Get the current web state of the spider
        spider_web_state = self.spiders[0].web_state

        # Convert the observations into integer indices
        observation = [spider_position[0], spider_position[1], spider_web_state]

        # Print the observation for debugging
        print(f"Observation: {observation}")

        return observation

    def is_valid_action(self, action):
        # Check if the action is a valid action for the spider
        x_action, y_action, web_action = action
        if x_action not in [0, 1, 2] or y_action not in [0, 1, 2] or web_action not in [0, 1]:
            return False
        return True

    def update_state(self, action):
        # Validate the action
        if not self.is_valid_action(action):
            raise ValueError("Invalid action")

        # Update the position of the spider and the state of the web based on the action
        self.spiders[0].move(action, self.webs[0])

        # If the spider is spinning a web, add its current position to the web's cells
        if self.spiders[0].web_state == 1:
            self.webs[0].update_state(self.spiders[0].position, self.grid_size)

        # If the spider is on a cell that is part of the web, update the spider's web state
        if self.spiders[0].position in self.webs[0].cells:
            self.spiders[0].web_state = 1
        else:
            self.spiders[0].web_state = 0

    def visualize(self):
        # Create a figure and axes
        fig, ax = plt.subplots()

        # Draw grid lines
        ax.set_xticks(np.arange(0, self.grid_size, 1))
        ax.set_yticks(np.arange(0, self.grid_size, 1))
        ax.grid(True)

        # Set the limits of the plot to the size of the grid
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)

        # Plot the spiders
        for spider in self.spiders:
            ax.plot(spider.position[1], spider.position[0], 'bo', markersize=10)  # 'bo' means blue circle

        # Plot the webs
        for web in self.webs:
            for cell in web.cells:
                ax.plot(cell[1], cell[0], 'rx')  # 'rx' means red cross

        # Show the plot
        plt.show()

class ActiveInferenceModel:
    def __init__(self):
        # Define the number of states for each factor
        self.num_states = num_states  # [number of x-coordinates (0-9), number of y-coordinates (0-9), web state (web, no web)]

        # Define the number of controls for each factor
        self.num_controls = num_controls  # [number of x-movements (move left, move right, stay), number of y-movements (move up, move down, stay), web-spinning action (spin web, don't spin web)]

        # Define the number of factors
        self.num_factors = num_factors  # [x-coordinate, y-coordinate, web state]

    def create_A(self, num_states):
        # Initialize the A matrix (observation likelihoods)
        A = utils.initialize_empty_A(num_obs, num_states)

        # The spider observes its x-coordinate and y-coordinate with perfect accuracy
        A[0] = np.eye(num_states[0])  # x-coordinate
        A[1] = np.eye(num_states[1])  # y-coordinate

        # The spider observes its web state with some error
        A[2] = np.array([
            [0.9, 0.1],  # If the true state is 0 (not on a web), the spider observes 0 with probability 0.9 and 1 with probability 0.1
            [0.1, 0.9]   # If the true state is 1 (on a web), the spider observes 1 with probability 0.9 and 0 with probability 0.1
        ])

        return A

    def create_B(self, num_states, num_controls):
        # Check if the number of states and controls are valid
        assert isinstance(num_states, list) and all(isinstance(n, int) and n > 0 for n in num_states), "num_states must be a list of positive integers"
        assert isinstance(num_controls, list) and all(isinstance(n, int) and n > 0 for n in num_controls), "num_controls must be a list of positive integers"
        assert len(num_states) == len(num_controls), "num_states and num_controls must have the same length"
        
        # Initialize the B matrices (transition matrices)
        B = utils.initialize_empty_B(num_states, num_controls)

        # Loop over each factor (x and y coordinates, and web state)
        for f, num_state in enumerate(num_states):

            if f < 2:  # f= 0 for (x-cordinate), f=1 for (y-coordinate)

                # MOVE LEFT (for x-coordinate) or UP (for y-coordinate)
                B[f][range(num_state-1), range(1, num_state), 0] = 1.0
                # If the spider is at leftmost state (state 0 for x) or topmost state (state 0 for y), Stay there
                B[f][0, 0, 0] = 1.0

                # MOVE RIGHT(for x-coordinate) or DOWN (for y-coordinate)
                B[f][range(1, num_state), range(num_state-1), 1] = 1.0
                # If the spider is at rightmost state (num_state-1 for x) or bottommost state (state num_state-1 for y), Stay there
                B[f][num_state-1, num_state-1, 1] = 1.0

                # STAY in the same position
                B[f][range(num_state), range(num_state), 2] = 1.0
            else:  # web state
                B[f][0, :, 0] = 1.0  # No web action keeps state as 'no web'
                B[f][1, :, 1] = 1.0  # Spin web action changes state to 'web'
        
        return B

    def create_C(self, num_obs):
        # Initialize the C matrix (preferences)
        C = utils.obj_array_zeros(num_obs)

        # The spider prefers to be on a web and in the center of the grid
        C[0][5] = 1.0  # Preference for x = 5
        C[1][5] = 1.0  # Preference for y = 5
        C[2][1] = 1.0  # Preference for web state = 1 (on a web)

        return C

    def create_D(self, num_states):
        # Initialize the D matrix (prior beliefs)
        D = [np.zeros(n) for n in num_states]

        # The spider initially believes it's not on a web and in the center of the grid
        D[0][5] = 1.0  # Initial belief for x = 5
        D[1][5] = 1.0  # Initial belief for y = 5
        D[2][0] = 1.0  # Initial belief for web state = 0 (not on a web)

        return np.array(D, dtype=object)
    
    def create_agent(self):
        A = self.create_A(num_states)
        B = self.create_B(num_states, num_controls)
        C = self.create_C(num_obs)
        D = self.create_D(num_states)

        return Agent(A=A, B=B, C=C, D=D)

# Define the size of the grid. This is how we define the size of the environment.
grid_size = 10

# Define the state space in the generative model. Cues from ActInf Modelstream #007.2
num_states = [grid_size, grid_size, 2]  # [number of x-coordinates, number of y-coordinates, web state (web, no web)]
num_factors = len(num_states)  # The number of factors in the generative model = 3

# Define the number of controls for each factor
num_controls = [3, 3, 2]  # [number of x-movements (move left, move right, stay), number of y-movements (move up, move down, stay), web-spinning action (spin web, don't spin web)]
num_controls_factors = len(num_controls)  # The number of control factors in the generative model = 3

# Define the number of possible observations for each observation modality
num_obs = [num_states[0], num_states[1], num_states[2]]  # 10 observations for x, 10 for y, 2 for web
# Create an instance of ActiveInferenceModel
model = ActiveInferenceModel()

# Create the agent
my_agent = model.create_agent()

# Initialize the environment
env = Environment(grid_size=10)
spider = Spider(start_position=(5, 5))
web = Web(center=(5, 5))
web.initialize_web_layout(grid_size)
env.add_spider(spider)
env.add_web(web)

for t in range(50):
    # The agent makes an observation of the environment
    observation = env.observe()
    # Print the type of the observation for debugging
    print(f"Type of observation: {type(observation)}")

    # # Convert the observation to a 2D array with shape (num_modalities, num_states)
    # num_states = [10, 10, 2]  # Replace this with the actual number of states for each modality
    # observation_array = np.zeros((len(observation), max(num_states)))
    # for i, obs in enumerate(observation):
    #     observation_array[i, obs] = 1

    # # Convert the observation_array to an integer array
    # observation_array = observation_array.astype(int)

    # Pass the converted observation to the infer_states method
    qs = my_agent.infer_states(observation)

    # The agent infers the optimal policy and samples an action from this policy
    q_pi, _ = my_agent.infer_policies()
    action = my_agent.sample_action()

    # The agent performs the action, which updates the state of the environment
    env.update_state(action)

    # Visualize the environment
    env.visualize()

    # Pause for a short period of time and then clear the figure
    plt.pause(0.1)
    plt.clf()