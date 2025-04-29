# AI Pacman Project (Reinforcement Learning Agents)

This project involves the implementation and testing of several reinforcement learning algorithms, including Value Iteration and Q-Learning, applied to various environments like Gridworld, a Crawler robot, and Pacman.

## Project Overview

The goal of this project was to implement key reinforcement learning algorithms from scratch and evaluate their performance in different scenarios. This includes both model-based (Value Iteration) and model-free (Q-Learning) approaches. Agents were developed to learn optimal policies for maximizing rewards in environments with varying dynamics, rewards, and state spaces.

## Algorithms Implemented

The core implementations can be found in the following files:
* `valueIterationAgents.py`: Contains implementations for model-based planning algorithms.
    * **Value Iteration (Batch):** Computes state values iteratively based on a known MDP model.
    * **Asynchronous Value Iteration:** Updates state values one state at a time per iteration cyclically.
    * **Prioritized Sweeping Value Iteration:** Focuses value updates on states where the value estimate change is significant, using a priority queue.
* `qlearningAgents.py`: Contains implementations for model-free learning algorithms.
    * **Q-Learning:** Learns action values (Q-values) directly from experience through interactions with the environment.
    * **Epsilon-Greedy Exploration:** Balances exploration (random actions) and exploitation (choosing the best-known action).
    * **Approximate Q-Learning:** Uses feature-based representations of states and actions to learn weights, enabling generalization across large state spaces (specifically for Pacman).
* `analysis.py`: Contains analysis and parameter settings for specific scenarios explored in the project questions.

## Environments

The agents were tested on several environments:

* **Gridworld:** Various grid-based worlds with different reward structures, transition noise, and layouts (e.g., BridgeGrid, DiscountGrid, MazeGrid) used to test fundamental algorithm correctness and parameter sensitivity.
* **Crawler Robot:** A simulated robot controller where the agent learns to move by controlling leg joints.
* **Pacman:** The classic game adapted as an MDP, where agents learn to eat food, avoid ghosts, and maximize score. Tested on small, medium, and classic layouts.

## How to Run

*(Note: Ensure you have Python installed and necessary libraries, if any, beyond the standard library. The project files assume a specific structure provided by the original assignment.)*

**Running Gridworld Examples:**

* **Value Iteration:**
    ```bash
    python gridworld.py -a value -i 100 -k 10
    ```
    * `-a value`: Use the ValueIterationAgent.
    * `-i 100`: Run 100 iterations of value iteration planning.
    * `-k 10`: Run 10 episodes using the learned policy.

* **Asynchronous Value Iteration:**
    ```bash
    python gridworld.py -a asynchvalue -i 1000 -k 10
    ```

* **Prioritized Sweeping:**
    ```bash
    python gridworld.py -a priosweepvalue -i 1000
    ```

* **Q-Learning:**
    ```bash
    python gridworld.py -a q -k 100 -m  # Run 100 learning episodes with manual control initially
    python gridworld.py -a q -k 100     # Run 100 learning episodes
    python gridworld.py -a q -k 100 --noise 0.0 -e 0.1 # Adjust noise and epsilon
    ```
    * `-a q`: Use the QLearningAgent.
    * `-k 100`: Run 100 episodes (learning occurs during these).
    * `-m`: Allow manual control (press keys to control agent).
    * `--noise`: Set transition noise probability.
    * `-e`: Set epsilon for epsilon-greedy exploration.
    * `-l`: Set the learning rate (alpha).

**Running the Crawler:**

```bash
python crawler.py
