import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# A. OpenAI Gym Overview (Printed Theory)
print("""
OpenAI Gym is a toolkit for reinforcement learning.
Environment: The world where agent acts.
Agent: Learner/decision maker.
Action: Move taken by agent.
Observation: State returned by environment.
Reward: Feedback signal.
""")

# B. Custom Maze Environment
class MazeEnv(gym.Env):

    def __init__(self):

        super(MazeEnv, self).__init__()

        self.size = 5

        self.maze = np.zeros((self.size, self.size))

        # Walls
        self.maze[1, 1] = -1
        self.maze[2, 2] = -1
        self.maze[3, 1] = -1

        self.start = (0, 0)
        self.goal = (4, 4)

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Discrete(self.size * self.size)

        self.reset()

    def reset(self, seed=None, options=None):

        self.agent_pos = self.start

        return self._get_state(), {}

    def _get_state(self):

        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):

        row, col = self.agent_pos

        if action == 0: row -= 1
        elif action == 1: row += 1
        elif action == 2: col -= 1
        elif action == 3: col += 1

        # Check boundaries
        if (0 <= row < self.size and
            0 <= col < self.size and
            self.maze[row, col] != -1):

            self.agent_pos = (row, col)

        reward = -1

        done = False

        if self.agent_pos == self.goal:

            reward = 100
            done = True

        return self._get_state(), reward, done, False, {}

    def render(self):

        grid = np.copy(self.maze)

        grid[self.goal] = 2
        grid[self.agent_pos] = 1

        print(grid)

# C & D. Q-Learning
env = MazeEnv()

state_size = env.observation_space.n
action_size = env.action_space.n

Q_table = np.zeros((state_size, action_size))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500
rewards = []

# E. Training Agent
for episode in range(episodes):

    state, _ = env.reset()

    total_reward = 0

    done = False

    while not done:

        if np.random.rand() < epsilon:

            action = env.action_space.sample()

        else:

            action = np.argmax(Q_table[state])

        next_state, reward, done, _, _ = env.step(action)

        Q_table[state, action] = (
            Q_table[state, action]
            + alpha * (
                reward + gamma * np.max(Q_table[next_state])
                - Q_table[state, action]
            )
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards.append(total_reward)

print("\nTraining Completed!")

# Plot Learning Progress
plt.plot(rewards)
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# F. Evaluation
success = 0
test_episodes = 50

for _ in range(test_episodes):

    state, _ = env.reset()

    done = False

    while not done:

        action = np.argmax(Q_table[state])
        state, reward, done, _, _ = env.step(action)

    if reward == 100:
        success += 1

print("\nSuccess Rate:",
      success / test_episodes * 100, "%")

# Visualize Learned Path
print("\nAgent Path After Training:")

state, _ = env.reset()
env.render()

done = False

while not done:

    action = np.argmax(Q_table[state])
    state, reward, done, _, _ = env.step(action)
    env.render()

print("\nMaze Solved ✅")