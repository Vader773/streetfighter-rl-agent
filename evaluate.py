from callback import TrainAndLoggingCallback, CHECKPOINT_DIR
from stable_baselines3 import PPO # Importing Proximal Policy Optimization (PPO) algorithm for our model
from stable_baselines3.common.evaluation import evaluate_policy # This helps us evaluate our model during hyperparam tuning to find the best one
from stable_baselines3.common.monitor import Monitor # This helps us monitor our training and log results. Monitor is a great way to get logging mean episode/reward values from wrapped/multiple parallel streams.
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # This helps us stack 4 frames together to give the ai a sense of time or trajectory.
from setup_env import StreetFighter
from model import LOG_DIR, OPT_DIR
import time

# Creating environment
env = StreetFighter()
env = Monitor(env, LOG_DIR) #Wrapped our custom env into monitor, which helps us properly recieve mean episode reward and shitmean episode length, and it logs those into the LOG_DIR folder.
env = DummyVecEnv([lambda: env]) # What does this do??
env = VecFrameStack(env, 4, channels_order='last') # From our env, we are gonna stack 4 diffrerent frames together, and as we had setup channels we can setup the order of those preprocessed 'channels' frames to last (i think?)


model = PPO.load('./opt/trial_26_best_model.zip', env=env) # Load the TRAINED checkpoint


# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1, render=True )

#------------------CUSTOM TESTING LOOP----------------------
#-------------TEST GAME LOOP-----------------
obs = env.reset() # Resets env and returns initial observation
done=False # Tells us if we have died in game or game is finished, so set to false initially
total_reward = 0
for game in range(1): # Loop through ONE game
    while not done: #Gonna loop through until we die
        if done:
            obs = env.reset() #Resets game if we die
        env.render() #Renders the game window

        action = model.predict(obs)[0] # Passing obs to model and making it make a prediction/action. Getting the first result with 0 so that its vectorized now?
        obs, reward, done, info = env.step(action) # passing action
        total_reward += reward
        # obs is observation after taking action, reward is the prevbuilt reward function to give reward, info gives valuable information and for step we tell the agent what to do after check obs
        time.sleep(0.01) # Sleep to slow down game a bit so its not too fast when rendering
        if reward != 0:
            print("--------------------")
            print("Current Game Score: ", info[0]['score']) # Adding [0] to all of these cos or else its a vector value and crashes the code. This happens cos we wrap the env in a DummyVecEnv
            print("Enemy Health:", info[0]['enemy_health'])
            print("Player Health:", info[0]['health'])
            print("Model Reward: ", reward)
            print("Total Model Reward: ", total_reward)


env.close() # Close the env after use to avoid clashes