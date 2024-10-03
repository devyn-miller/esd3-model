# Configuration settings for the training process

# Environment parameters
CUSTOMERS = 3000
PERIODS = 100
STARTING_PRICES = [300, 300, 300]
LARGEST_DISCOUNT = 30  # most you can decrease your prices in one period
LARGEST_INCREASE = 30   # most you can increase your prices in one period
# INERTIA = min(prices)/10
INERTIA = 10**10

# Training parameters
NUM_AGENTS = 4
TIMESTEPS_TOTAL = 500000
MAX_TRAINING_ITERATION = 10000