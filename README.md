# Introduction
Developed a (simplified) Branching-Time Active Inference (BTAI) framework, integrating Active
Inference with Monte Carlo Tree Search (MCTS) for sequential decision-making under
uncertainty. Implemented and tested the model using the PyMDP library.

The work draws from [here](https://www.sciencedirect.com/science/article/pii/S0893608022001149).

To run the Agent the [PyMDP](https://github.com/infer-actively/pymdp) package is required.

## Example
The Example folder contains three examples based on the main environment described by the PyMDP examples page:
  - **GridWorld**: a nxn 2D space where the agents needs to reach and end point
  - **T-Maze**: a T-Maze where the agent is required to gather evidence in order to obtain a reward (food) instead of a punishment (shock).
  - **Epistemic chaining**: an extend version of the T-Maze, which include two evidence steps required to get the reward.

