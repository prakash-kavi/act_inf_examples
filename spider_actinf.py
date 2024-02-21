import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict

import sys
sys.path.append(r'E:\phd-proj\dth\pymdp\pymdp')

from pymdp import utils
from pymdp.agent import Agent
from pymdp.maths import softmax

global grid_size

class Spider:  
    def __init__(self, start_position, web_state=0):
        self.position = start_position  # The spider's current position
        self.web_state = web_state  # The spider's web state
        self.trace = [start_position]  # The trace of the spider's route

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        assert isinstance(value, tuple) and len(value) == 2, "Position must be a tuple of length 2"
        self._position = value

    def move(self, action, web):
        x_action, y_action, web_action = action

        # Add randomness to the action state
        x_action = np.random.choice([0, 1, 2])
        y_action = np.random.choice([0, 1, 2])
        web_action = np.random.choice([0, 1])

        new_position = list(self.position)
        if x_action == 0 and self.position[1] > 0:  # Move left
            new_position[1] -= 1
        elif x_action == 1 and self.position[1] < grid_size - 1:  # Move right
            new_position[1] += 1
        if y_action == 0 and self.position[0] > 0:  # Move up
            new_position[0] -= 1
        elif y_action == 1 and self.position[0] < grid_size - 1:  # Move down
            new_position[0] += 1

        # Check if the new position is part of the web layout
        if tuple(new_position) in web.cells:
            # Update the spider's position
            self.position = tuple(new_position)
            self.trace.append(self.position)  # Add the new position to the trace

        # If the spider is spinning a web and it's not already on a web cell, set the web state to 1
        if web_action == 1 and self.position not in web.cells:
            self.web_state = 1
        else:  # Stop spinning web
            self.web_state = 0

class Web:
    def __init__(self, center):
        self.center = center  # The center of the web
        self.cells = []  # The cells that are part of the web

    def initialize_web_layout(self, grid_size):
            # Build the web in a circular style with radius 2
            for dx in range(-2, 3):  # dx is the change in x-coordinate
                for dy in range(-2, 3):  # dy is the change in y-coordinate
                    if dx**2 + dy**2 <= 4:  # Check if the cell is within a distance of 2 from the center
                        x = self.center[0] + dx
                        y = self.center[1] + dy
                        if 0 <= x < grid_size and 0 <= y < grid_size:
                            self.cells.append((x, y))  # Add the cell to the web layout

            self.cells = list(set(self.cells))  # Remove duplicates

    def update_state(self, action):
        # Validate the action
        if not self.is_valid_action(action):
            raise ValueError("Invalid action")

        # Add randomness to the action state
        action = (np.random.choice([0, 1, 2]), np.random.choice([0, 1, 2]), np.random.choice([0, 1]))

        # Move the spider based on the action
        self.spiders[0].move(action, self.webs[0])

        # Update the spider's web state based on its position
        if self.spiders[0].position in self.webs[0].cells:
            self.spiders[0].web_state = 1  # Set web state to 1 if spider is on a cell that is part of the web

class Environment:
    def __init__(self, grid_size):
        self.grid_size = grid_size      # The size of the grid in the environment
        self.spiders = []               # The spiders in the environment
        self.webs = []                  # The webs in the environment

    def add_spider(self, spider):
        # Add a spider to the grid
        self.spiders.append(spider)

    def add_web(self, web):
        # Add a web to the grid
        self.webs.append(web)

    def observe(self):
        # Returns the current position and web state of the spider as an observation.
        spider_position = self.spiders[0].position       # Get the current position of the spider
        spider_web_state = self.spiders[0].web_state           # Get the current web state of the spider

        # Convert the observations into integer indices
        observation = [spider_position[0], spider_position[1], spider_web_state]

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

        # Move the spider based on the action
        self.spiders[0].move(action, self.webs[0])

        # Update the spider's web state based on its position
        if self.spiders[0].position in self.webs[0].cells:
            self.spiders[0].web_state = 1  # Set web state to 1 if spider is on a cell that is part of the web
    
    def visualize(self):
        # Create a grid to represent the environment
        grid = np.zeros((self.grid_size, self.grid_size))

        # Mark the cells that are part of the web
        for cell in self.webs[0].cells:
            grid[cell] = 2 if cell in self.spiders[0].trace else 1

            # Mark the current position of the spider
            grid[self.spiders[0].position] = 3

            # Plot the grid
            plt.imshow(grid, cmap='viridis')
            plt.draw()

class ActiveInferenceModel:
    def __init__(self, web, spider):
        self.web = web
        self.spider = spider
        # Define the number of states for each factor
        self.num_states = num_states  # [number of x-coordinates (0-9), number of y-coordinates (0-9), web state (web, no web)]
        # Define the number of controls for each factor
        self.num_controls = num_controls  # [number of x-movements (move left, move right, stay), number of y-movements (move up, move down, stay), web-spinning action (spin web, don't spin web)]
        # Define the number of factors
        self.num_factors = num_factors  # [x-coordinate, y-coordinate, web state]
    
    def update_spider_position(self, action):
        self.spider.move(action, self.web)

    def create_A(self, num_states):
        # Constructs the 'A' matrix, representing the spider's high-fidelity proprioceptive sensory observations about grid position and web state - 1 modality.

        # Calculate the total number of observations: 10 * 10 * 2 = 200
        total_num_obs = np.prod(num_states)

        # Initialize the A matrix (observation likelihoods)
        A = utils.initialize_empty_A([total_num_obs], num_states)
        A[0] = np.zeros([total_num_obs] + num_states)

        # Loop over the Y coordinates
        for y in range(num_states[1]):
            # Loop over the X coordinates
            for x in range(num_states[0]):
                # Loop over the web presence (0 for no web, 1 for web)
                for w in range(num_states[2]):
                    # Calculate the index into the observation array
                    obs_index = (y * num_states[0] + x) * num_states[2] + w
                    # Fill out the mapping to observations for the possible settings of X, Y, and W
                    A[0][obs_index, x, y, w] = 1

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

    def create_C(self, num_states):
        total_num_obs = np.prod(num_states)  # Total number of observations = 10*10*2 = 200
        current_position = self.spider.position

        # Initialize the C matrix (preferences)
        C = utils.obj_array_zeros([total_num_obs])

        # The spider prefers to be on a cell that is part of the web layout and does not have a web
        for x in range(num_states[0]):
            for y in range(num_states[1]):
                for w in range(num_states[2]):
                    # Calculate the index into the observation array
                    obs_index = (y * num_states[0] + x) * num_states[2] + w
                    if (x, y) in self.web.cells:
                        if w == 0:
                            C[0][obs_index] = 2.0  # Preference for being on a cell that is part of the web layout and does not have a web
                        else:
                            C[0][obs_index] = 0.1  # Lower preference for staying in a position where a web has already been spun
                    else:
                        C[0][obs_index] = 0.0  # No preference for cells that are not part of the web layout

                    # Distance-based preference
                    distance = abs(x - current_position[0]) + abs(y - current_position[1])
                    C[0][obs_index] += distance / (num_states[0] + num_states[1])

                    # Time-based preference
                    C[0][obs_index] -= self.spider.trace.count((x, y, w)) / total_num_obs

                    # Randomness
                    C[0][obs_index] += np.random.normal(0, 0.1)

        return C

    def create_D(self, num_states):
        num_factors = len(num_states)
        # Initialize D with an object array of size num_factors
        D = utils.obj_array(num_factors)

        # Initialize D[0] and D[1]
        for i in range(2):
            D[i] = np.full(num_states[i], 0.1)  # Initialize with 0.1 for all states
            D[i][5] = 0.5  # Modest prior for the center cell
            D[i] = D[i] / D[i].sum()  # Normalize D[i]

        # Initialize D[2]
        D[2] = np.full(num_states[2], 0.5)  # No preference for the web state
        D[2] = D[2] / D[2].sum()  # Normalize D[2]

        print(f"D length: {len(D)}")  # Print the length of D
        print(f"D shapes: {[d.shape for d in D]}")  # Print the shapes of the arrays in D
        return D
    
    def create_agent(self):
        A = self.create_A(num_states)
        B = self.create_B(num_states, num_controls)
        C = self.create_C(num_states)
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

# Initialize the spider and the web
spider = Spider(start_position=(5, 5))
web = Web(center=(5, 5))
web.initialize_web_layout(grid_size)

# Create an instance of ActiveInferenceModel
model = ActiveInferenceModel(web, spider)

# Create the agent
my_agent = model.create_agent()

# Initialize the environment
env = Environment(grid_size=10)
env.add_spider(spider)
env.add_web(web)

# Create the figure
plt.figure()

# Initialize a variable to keep track of whether the spider has just spun a web
just_spun_web = False

for t in range(20):
    # Clear the figure for the next iteration
    plt.clf()

    # The agent makes an observation of the environment
    observation = env.observe()
    print(f"Observation: {observation}")

    # The agent infers the optimal policy and samples an action from this policy
    qs = my_agent.infer_states(observation)
    q_pi, _ = my_agent.infer_policies()
    action = my_agent.sample_action()

    # Update the spider's position
    model.update_spider_position(action)

    # Update the preferences
    model.C = model.create_C(num_states)

    # The agent performs the action, which updates the state of the environment
    env.update_state(action)
    # print(f"Spider's trace: {env.spiders[0].trace}")

    # Visualize the environment
    env.visualize()

    # Pause for a longer period of time to slow down the plot updates
    plt.pause(0.5)