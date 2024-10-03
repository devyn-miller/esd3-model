# Customer Inertia and Price Insensitivity Simulation

## Overview

This project aims to model customer inertia and price insensitivity in subscription-based industries. The goal is to develop an optimal pricing strategy using reinforcement learning.

## Objectives

1. Conduct a literature review on customer inertia and price insensitivity.
2. Develop a mathematical model based on the literature.
3. Implement the model in a reinforcement learning environment.
4. Train and evaluate the model to determine optimal pricing strategies.

## Directories and Files

- **/src/**: Main source directory containing all the code.

   - **/models/**: Contains mathematical models related to the project.
      - `market_model.py`: Implements the `calculate_market_share` function and other relevant models.

   - **/env/**: Contains the environment setup for reinforcement learning.
      - `inertia_env.py`: Implements the `InertiaEnv` class and related logic.

   - **/training/**: Contains scripts for training the model.
      - `train.py`: Handles the training process and execution of the reinforcement learning model.
      - `config.py`: Contains configuration settings for training, such as hyperparameters.

   - **/utils/**: Contains utility functions for reuse across the project.
      - `helpers.py`: Includes helper functions for various tasks.

   - `main.py`: The main entry point for running the simulation or training.

- **/data/**: Directory for storing data-related files.

   - **results/**: Stores results from training or simulations.
   - **logs/**: Stores log files for tracking the training process.

- **/tests/**: Directory for unit tests.

   - `test_market_model.py`: Tests for functions in `market_model.py`.
   - `test_inertia_env.py`: Tests for the `InertiaEnv` class.
   - `test_training.py`: Tests for training scripts.

- **requirements.txt**: Lists all dependencies required for the project.
- **README.md**: Provides an overview of the project.
- **.gitignore**: Specifies files and directories to be ignored by version control.
- **RL.ipynb**: Jacob's initial reinforcement learning for customer inertia model.

## Usage

### Running the Simulation

1. Modify the `InertiaEnv` class to incorporate the new mathematical model.
2. Configure the PPO algorithm in the `RL.ipynb` file.
3. Run the training script:

**note: I will likely be using RL.py to run the simulation because Jupyter is very computationally expensive for my computer.**

### Evaluating Results

- After training, evaluate the model's performance by analyzing the reward metrics.
- Adjust the model and environment setup based on the evaluation.

## Code Structure

- `RL.ipynb`: Main script containing the environment and training setup.
- `calculate_market_share`: Function to calculate market share based on prices.
- `InertiaEnv`: Custom environment class for simulating customer inertia.

## References

- [Cao, Manthiou, and Ayadi (2022)](https://doi.org/10.1016/j.jbusres.2022.02.013)
- [Miller, Sahni, and Strulov-Shlain (2022)](https://doi.org/10.2139/ssrn.4065098)
- [Su (2008)](https://doi.org/10.2139/ssrn.945903)