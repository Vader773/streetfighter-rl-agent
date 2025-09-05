# Importing retro to setup env and play game in python
import retro
import numpy as np
import time # Timported time to slow down game a bit so its not too fast when rednering
import cv2
import matplotlib.pyplot as plt
from setup_env import StreetFighter # Importing our custom environment class from setup_env.py

env = StreetFighter() # Creates an instance of our custom environment (No output means no error!)

obs = env.reset() # Resets env and returns initial observation

for plots in range(10): # Loop through 10 plots
    obs, reward, done, info = env.step(env.action_space.sample()) # We take a sample random action in the game
    plt.imshow(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)) # We use cvtCOLOR to convert BGR to RGB as matplotlib uses RGB format
    plt.show()