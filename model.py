# Default configuration
CONFIG = {
    'N_TRIALS': 10,
    'TOTAL_TIMESTEPS': 30000,
    'N_EVAL_EPISODES': 5,
    'VERBOSE': True,
    'LOG_DIR': './logs/',
    'OPT_DIR': './opt/'
}

# Try to import external config to override defaults
try:
    from config import CONFIG as EXTERNAL_CONFIG
    CONFIG.update(EXTERNAL_CONFIG)
    print(f"📝 External configuration loaded and merged!")
except ImportError:
    print(f"📝 Using default configuration")

# Extract variables for backward compatibility
N_TRIALS = CONFIG['N_TRIALS']
TOTAL_TIMESTEPS = CONFIG['TOTAL_TIMESTEPS']
N_EVAL_EPISODES = CONFIG['N_EVAL_EPISODES']
VERBOSE = CONFIG['VERBOSE']
LOG_DIR = CONFIG['LOG_DIR']
OPT_DIR = CONFIG['OPT_DIR']

import optuna # optimization import which optimizes hyperparams and trains at the same time
from stable_baselines3 import PPO # Importing Proximal Policy Optimization (PPO) algorithm for our model
from stable_baselines3.common.evaluation import evaluate_policy # This helps us evaluate our model during hyperparam tuning to find the best one
from stable_baselines3.common.monitor import Monitor # This helps us monitor our training and log results. Monitor is a great way to get logging mean episode/reward values from wrapped/multiple parallel streams.
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # This helps us stack 4 frames together to give the ai a sense of time or trajectory.
from setup_env import StreetFighter
import os
import traceback
import joblib
import shutil


#---------FUNCTIONS------------

# Function to return tested hyperparameters- defining the object function which lets us suggest which hyperparams optuna should tune and how much 
def optimize_ppo(trial):
    return {
        # optuna works like this: trial.suggest[type of output[int/loguniform/uniform/etc [what are those?]]]([param we want to train], min value, max value)
        'n_steps': trial.suggest_int('n_steps', 2048, 8192), # No.of frames that are stored and used for ONE batch of training our PPO model
        'gamma': trial.suggest_float('gamma', 0.8, 0.9999, log=True),# Future rewards that are calculated as part of the ppo model
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),  # How fast or slow we tune our optimizer
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4), # How much we want to clip the advantage value (CHECK MORE ON THIS?)
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99) # Smoothing param (CHECK MORE ON THIS?)
    }

# Function to run a training loop and return mean reward values
def optimize_agent(trial):
    try:
        if VERBOSE:
            print(f"\nStarting Trial {trial.number + 1}/{N_TRIALS}") # Cos trail number will be 0 initially so its a bit confusing
        
        model_params = optimize_ppo(trial)

        # Creating environment
        env = StreetFighter()
        env = Monitor(env, LOG_DIR) #Wrapped our custom env into monitor, which helps us properly recieve mean episode reward and shitmean episode length, and it logs those into the LOG_DIR folder.
        env = DummyVecEnv([lambda: env]) # What does this do??
        env = VecFrameStack(env, 4, channels_order='last') # From our env, we are gonna stack 4 diffrerent frames together, and as we had setup channels we can setup the order of those preprocessed 'channels' frames to last (i think?)

        #---------------CREATING THE PPO MODEL-----------------
        
        """Here, we are using a PPO RL model and a CNN policy cos the model reads from fram delta (images), 
        which will be working through our custom env,
        and tensorboard logs will be saved in the LOG_DIR folder that we set above,
        we dont want verbose or else it just spams the terminal and prints a shit ton of info, so set verbose to 0, 
        and THE MOST IMPORTANT BIT, we are unpacking the param values from optuna that we get in the above function and passing it while creating the ppo model as its params."""

        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params) 

        print(f"Training for {TOTAL_TIMESTEPS} timesteps")

        model.learn(total_timesteps=TOTAL_TIMESTEPS) #The model will train for timesteps mentioned in config

        #---------------EVALUATING THE MODEL-------------------

        # Here, "mean)reward, _" means we are UNPACKING the results we got from our policy before passing it to evaluate function
        print(f"Evaluating model for {N_EVAL_EPISODES} games.")
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)  # Evaluation of our model, through our environment, for n number of GAMES to play
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number)) # Saved to our path OPT_DIR, and then formatted name such that trial number is put in place of {}
        model.save(SAVE_PATH, include_env=True) # IMP: Saves our model. Include env includes our obs/action space for later reloading the model.

        if VERBOSE:
            print(f" Trial: {trial.number + 1}")
            print(f" Mean Reward: {mean_reward:.2f}")
            print(f" Model Saved: {SAVE_PATH}")

        return mean_reward

    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed due to: {e}")
        traceback.print_exc()
        return -1000 # give placeholder value so that entire loop doenst break in case of one or two errors.

SAVE_EVERY = 10  # save db every 2 trials (adjust as needed)
# Code from chatgpt to save the study periodically
def periodic_checkpoint(study, trial):
    if trial.number % SAVE_EVERY == 0:
        os.makedirs("/kaggle/working/opt", exist_ok=True)
        shutil.copy2("optuna_study.db", "/kaggle/working/opt/optuna_study.db")
        print(f"💾 [Optuna] Checkpoint saved after trial {trial.number}")
        
if __name__ == "__main__": #Wrapped in this loop so that it runs only when this file is executed....this is done to safely extract study to other files like train.py
    
    # Save study persistently to disk
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_study.db",  # SQLite file
        engine_kwargs={"connect_args": {"timeout": 10}}
    )
    
    # Creating the experiment (?)
    study = optuna.create_study(storage=storage, study_name="streetfighter", direction='maximize', load_if_exists=True) # It is important to set direction to MAXIMIZE if we want to return positive mean rewards, if not then we must return negative mean rewards (why?)

    # Running the study through our optimize agent function, where n_trials means HOW MANY DIFFERENT SETS of hyperparams we will be testing using optuna, and n_jobs specifies how many PARALLEL ENVIRONMENTS are used to train at the same time

    print(f"Starting optimization with {N_TRIALS} trials:")
    estimated_time = (TOTAL_TIMESTEPS * N_TRIALS) / 60000
    print(f"Estimated time: {estimated_time:.0f}-{estimated_time*2:.0f} minutes")

    study.optimize(optimize_agent, n_trials=N_TRIALS, n_jobs=1, callbacks=[periodic_checkpoint]) #Calling periodic checkpoint callback 
    # study.optimize(optimize_agent, n_trials=100, n_jobs=1) #For the actual hardcore training....estimated to take more than 22 hours!
    

# TODO : 
# We gotta finish this by tomorrow (9/5/25)