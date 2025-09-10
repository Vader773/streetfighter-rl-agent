"""
What we will be doing here.
We will PREPROCESS the game to make learning and playing easier for our agent
1) We will grayscale the game cos we don't need colours to play tha game.
2) We will downsize the game to 84x84 pixels cos we don't need a high res image to play the game, so now the game will be 84x84x1 (lengthxwidthxcolor)
3) We will capture the "frame delta". This means the current frame MINUS the previous frame. This lets the AI know about moves and movements
4) We will change the action parameters by filtering them so that only the VIABLE attack combinations are present at any given time instead of all 4096 possible actions. This will make learning easier for the AI
5) We will change and properly create our reward fuction (Make it proportional to ingame score taken from info?? Or maybe rewards for hitting enemy (calculating loss in enemy health taken from info!)) 
We will set reward function to score FOR NOW, and maybe if we tweak it up later on for rl vs rl, we can do enemy health taken = reward and player health gone = penalty 
"""


from gym import Env # Imported environment base class
from gym.spaces import MultiBinary, Box # Imported action space (which is MultiBinary)
import numpy as np # For DA MATH (frame delta)
import retro
import cv2 # For the image preprocessing like scaling down and greyscaling

#---------------CREATE CUSTOM ENVIRONMENT CLASS BOIS-----------------
# Custom environment will basically change the existing env to the one we want as mentioned above
class StreetFighter(Env): #Passed base environment (Env) to custom environment
    def __init__(self): #This function is a must i think....its the first thing that gets called when the environment is created so we put all variable we wanna start with and set them up here
        super().__init__() # This inherits our base environment from the Env that we have set as input
         
        #-------------------Specify our action and observation space--------------------

        #We are essentially recreated our original environment here as a start BEFORE preprocessing it later on!
        # Here, BOX is like an np array which can be set to any size as needed (we see that env.observation_space returns a box!), low and high are for colours, shape is the shape of our env and dtype is also the dtype we got from env.observation_space
        # Rechange: changed shape from (200, 256, 3) to (84, 84, 1) becoz we had to reshape it after grayscaling and resizing during preprocessing
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        # Again, we are recreating and KEEPING original action space for ai to use (got from env.action_space)
        self.action_space = MultiBinary(12) #12 buttons in street fighter, each can be 0 or 1 (not pressed or pressed)


        # Now, we will start a new instance of the game
        # We use self.game to make sure that we can use this variable in other functions of the class too, instead of self.env
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', 
                              use_restricted_actions=retro.Actions.FILTERED) #This makes sure that only the valid attack combination actions are present at any given type instead of all actions so that ai doesnt go spamming random nonsensical actions.
        

    def reset(self): #This function resets/restarts our environment for repeated use while training
        obs = self.game.reset() #Resets the game and returns the initial observation (initial game frame)

        obs = self.preprocess(obs) #Preprocess the initial observation (game frame) using our preprocess function defined below
        #----------FRAME DELTA-----------
        #frame delta = current frame - previous frame
        self.previous_frame = obs #Lets set previous preprocessed frame to initial frame for now 
        
        # Taking one dummy action which does nothing to access info and get current player and enemy health to initialise in reset
        dummy_action = [0] * self.action_space.n # No actions taken
        _, _, _, info = self.game.step(dummy_action)

        self.health = info['health']
        self.enemy_health = info['enemy_health']

        self.max_health = info['health']
        self.max_enemy_health = info['enemy_health']

        self.round_over = False

        return obs
    
    def preprocess(self, observation): # This function preprocesses the observation (game frame) to make it suitable for our agent
        # Here, we just need to grayscale and resize game to lower pixels. frame delta will be done in step function
        gray=cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) # This grayscales our entire screen.

        # Resizing
        resize = cv2.resize(gray, (84, 84), interpolation = cv2.INTER_CUBIC) #Calling grayscaling here and resizing game for less pixels
        
        #Ok, so we found out that making it (84, 84, 1) isnt viable, so we are gonna make it (84, 84), and then add a "channels" value below using numpy to fix everything
        # This is becoz our rl package, stable baselines, expects an obs in a 3D array like so: (length, width, channels)
        #Channels value
        channels = np.reshape(resize, (84, 84, 1)) # This adds a channels value to the end of the array to make it compatible with our observation space (which is 3D)

        return channels # Return the preprocessed frame
    
    def step(self, action): # This function is called when we want to take a step in the environment (i.e. take an action)
        #Take a step
        obs, reward, done, info = self.game.step(action) # We take the action in the game and get the new observation, reward, done and info

        #Processing our observations 
        obs = self.preprocess(obs)

        #===========FRAME DELTA================
        # frame_delta = obs - self.previous_frame # This calculates the frame delta (current frame - previous frame)
        # self.previous_frame = obs #So that when we take our NEXT step, we have the previous frame set to the current frame

        # Maybe its better to just pass the normall obs to the ai as frame delta isn't giving much info to the ai at all
        frame_delta = obs

        #===========REWARD FUNCTION=============
        # We can get game score from info dictionary

        # For round reset
        if info['health'] > self.health and info['enemy_health'] > self.enemy_health:
            # Defining max health for progressive round rewarding
            self.max_health = info['health']
            self.max_enemy_health = info['enemy_health']
            # Reseting the other info
            self.health = info['health']
            self.enemy_health = info['enemy_health']
            self.round_over = False
            return frame_delta, 0, done, info # 0 for reward for reset

        # If round is over, no reward
        if self.round_over:
            return frame_delta, 0, done, info
        
        if not self.round_over:
            # Main reward function
            reward = ((self.enemy_health - info['enemy_health']) * 10) - ((self.health - info['health']) * 10) # This sets reward to the CHANGE in enemy health - change in player health

            # Big reward for win and big penalty for loss, but if both end up with 0 health (Double KO or round reset), the no reward
            if info['health'] <= 0 or info['enemy_health'] <= 0:
                terminal_reward = round(((info['health'] - info['enemy_health']) / self.max_health) * 1000, 2)
                reward += terminal_reward
                self.round_over = True
            else:
                reward += 0

        self.enemy_health = info['enemy_health'] # This now updates the enemy health info stored to the new enemy health of the player in game
        self.health = info['health'] # Same for this

        return frame_delta, reward, done, info
    
    #for *args and **kwargs, we are just unpacking and arguments or keyword areguments from stable baslines (our rl package) so that it doesnt throw an error  
    def render(self, *args, **kwargs): # This makes sure HOW we go abt rendering the game and whether or not we render (not needed for headless envs ig)
        self.game.render() # Renders the game

    def close(self): # This function is called when we want to close the environment
        self.game.close() # Closes game

#-----------------TESTING OUR CUSTOM ENVIRONMENT-----------------
env = StreetFighter() # Creates an instance of our custom environment (No output means no error!)

obs = env.reset()      # reset environment to get initial frame
action = env.action_space.sample()  # take a random action
obs, reward, done, info = env.step(action)  # step once
print(info)  # Now this will work

print(env.observation_space.shape)
print(env.action_space.sample())
env.close()

# TODO: PROPER REWARD FUNCTION