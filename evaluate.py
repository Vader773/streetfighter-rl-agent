from callback import TrainAndLoggingCallback, CHECKPOINT_DIR
from stable_baselines3 import PPO # Importing Proximal Policy Optimization (PPO) algorithm for our model
from stable_baselines3.common.evaluation import evaluate_policy # This helps us evaluate our model during hyperparam tuning to find the best one
from stable_baselines3.common.monitor import Monitor # This helps us monitor our training and log results. Monitor is a great way to get logging mean episode/reward values from wrapped/multiple parallel streams.
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # This helps us stack 4 frames together to give the ai a sense of time or trajectory.
from setup_env import StreetFighter
from model import LOG_DIR, OPT_DIR

# Creating environment
env = StreetFighter()
env = Monitor(env, LOG_DIR) #Wrapped our custom env into monitor, which helps us properly recieve mean episode reward and shitmean episode length, and it logs those into the LOG_DIR folder.
env = DummyVecEnv([lambda: env]) # What does this do??
env = VecFrameStack(env, 4, channels_order='last') # From our env, we are gonna stack 4 diffrerent frames together, and as we had setup channels we can setup the order of those preprocessed 'channels' frames to last (i think?)


model = PPO.load('./train/best_20000_model.zip') # Load the TRAINED checkpoint

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, render=True )