# Importing retro to setup env and play game in python
import retro
import numpy as np
import time # Imported time to slow down game a bit so its not too fast when rednering
from setup_env import StreetFighter # Importing our custom environment class from setup_env.py


# List of all games in gym-retro:
# retro.data.list_games())

# We gotta run "python -m retro.import ." before starting in roms folder to import rom into retro

# env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis') # Gotta make sure this name is the same in list of all games

env = StreetFighter() # Creates an instance of our custom environment (No output means no error!)

print(env.action_space) #Shows action space, which is 12 in this case (2^12 = 4096 possible actions)


#-------------TEST GAME LOOP-----------------
obs = env.reset() # Resets env and returns initial observation
done=False # Tells us if we have died in game or game is finished, so set to false initially
for game in range(1): # Loop through ONE game
    while not done: #Gonna loop through until we die
        if done:
            obs = env.reset() #Resets game if we die
        env.render() #Renders the game window
        obs, reward, done, info = env.step(env.action_space.sample()) # We take a sample random action in the game
        # obs is observation after taking action, reward is the prevbuilt reward function to give reward, info gives valuable information and for step we tell the agent what to do after check obs
        time.sleep(0.0025) # Sleep to slow down game a bit so its not too fast when rendering
        if reward > 0:
            print("--------------------")
            print("Current Game Score: ", info['score'])
            print("Model Reward: ", reward)


env.close() # Close the env after use to avoid clashes

