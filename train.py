# Default configuration for local debugging. We will overwrite this configuration in kaggle cell by extracting it from a config file is available.
CONFIG = {
    'CHECK_FREQ' : 10000,
    'TOTAL_TIMESTEPS': 10000,
    'VERBOSE' : False,
    'INCREMENTAL_TRAINING': False
}

try:
    from config import CONFIG as EXTERNAL_CONFIG
    CONFIG.update(EXTERNAL_CONFIG)
    print(" External config found and overwritten")
except ImportError:
    print(" Using default config cos we are prolly in local")


# Extracting the dictionary values into python variables for easier accessibility across script
INCREMENTAL_TRAINING = CONFIG['INCREMENTAL_TRAINING']
VERBOSE = CONFIG['VERBOSE']
CHECK_FREQ = CONFIG['CHECK_FREQ']
TOTAL_TIMESTEPS = CONFIG['TOTAL_TIMESTEPS']


from callback import TrainAndLoggingCallback, CHECKPOINT_DIR
from stable_baselines3 import PPO # Importing Proximal Policy Optimization (PPO) algorithm for our model
from stable_baselines3.common.evaluation import evaluate_policy # This helps us evaluate our model during hyperparam tuning to find the best one
from stable_baselines3.common.monitor import Monitor # This helps us monitor our training and log results. Monitor is a great way to get logging mean episode/reward values from wrapped/multiple parallel streams.
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # This helps us stack 4 frames together to give the ai a sense of time or trajectory.
from setup_env import StreetFighter
from model import LOG_DIR, OPT_DIR
import os
import traceback
import optuna
import joblib


callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ, save_path=CHECKPOINT_DIR) # Every 10k steps, we will save and log the model!

# Copying the code from model.py

# Creating environment
env = StreetFighter()
env = Monitor(env, LOG_DIR) #Wrapped our custom env into monitor, which helps us properly recieve mean episode reward and shitmean episode length, and it logs those into the LOG_DIR folder.
env = DummyVecEnv([lambda: env]) # What does this do??
env = VecFrameStack(env, 4, channels_order='last') # From our env, we are gonna stack 4 diffrerent frames together, and as we had setup channels we can setup the order of those preprocessed 'channels' frames to last (i think?)



storage = optuna.storages.RDBStorage(
    url="sqlite:///optuna_study.db",
    engine_kwargs={"connect_args": {"timeout": 10}}
)

study = optuna.load_study(study_name="streetfighter", storage=storage)

if VERBOSE:
    # Get top 5 trials sorted by reward (descending)
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]

    print("\n🏆 Top 5 Trials:")
    for i, trial in enumerate(top_trials):
        print(f"\nRank {i+1}")
        print(f" Trial Number: {trial.number}")
        print(f" Mean Reward: {trial.value:.2f}")
        print(f" Parameters: {trial.params}")
        print(f" Model Path: {os.path.join(OPT_DIR, f'trial_{trial.number}_best_model')}")

    # Print best trial info (top 1)
    best_trial = study.best_trial
    print("\n🔥 Best Trial Overall:")
    print(f" Trial Number: {best_trial.number}")
    print(f" Mean Reward: {best_trial.value:.2f}")
    print(f" Parameters: {best_trial.params}")
    print(f" Model Path: {os.path.join(OPT_DIR, f'trial_{best_trial.number}_best_model')}")

# Access the best parameters

# study = joblib.load('ppo_study.pkl')

if not INCREMENTAL_TRAINING:
    # Get Trial 4 parameters (4th best model)
    trial_4 = next((trial for trial in study.trials if trial.number == 4), None)
    if trial_4 is None:
        raise ValueError("Trial 4 not found!")

    model_params = trial_4.params.copy()  # Use .copy() to avoid modifying original
    print("Original Trial 4 params:", model_params)

    model_params['n_steps'] = 2368  # Set to closest multiple of 64
    model_params['learning_rate'] = 5e-7  # Override learning rate
    print("Modified params:", model_params)

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params) 

    # Load weights into this new model
    model.set_parameters(os.path.join(OPT_DIR, 'trial_4_best_model.zip'))

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, reset_num_timesteps=False) # Use callback that we created

# model_params = study.best_params
# print(model_params)
# model_params['n_steps'] = 2368 # ALWAYS set n_steps to closest multiple of 64
# model_params['learning_rate'] = 5e-7
# print(model_params)

if INCREMENTAL_TRAINING:
    # WHEN DOING INCREMENTAL TRAINING
    model = PPO.load('./train/train_5_final_model.zip', env=env)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, reset_num_timesteps=False) # Use callback that we created

model.save(os.path.join(CHECKPOINT_DIR, "train_5_final_model"))

# If we need tensorboard, open new terminal and do cd logs --> tensorboard --logdir=.