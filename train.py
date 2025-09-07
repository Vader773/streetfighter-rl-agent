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


callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR) # Every 10k steps, we will save and log the model!

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

# Access the best parameters

# study = joblib.load('ppo_study.pkl')

model_params = study.best_params
print(model_params)
model_params['n_steps'] = 2304 # ALWAYS set n_steps to closest multiple of 64
model_params['learning_rate'] = 5e-7
print(model_params)

# WHEN WE ARE TRAINING PROPERLY, we will wanna do these:
# model.learn(total_timesteps=5000000)

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params) 


# Load weights into this new model
model.set_parameters(os.path.join(OPT_DIR, 'trial_3_best_model.zip'))

# KEY CHANGE: Loading our pre-existing best model 

model.learn(total_timesteps=5000000, callback=callback) # Use callback that we created

model.save(os.path.join(CHECKPOINT_DIR, "train_3_final_model"))

# If we need tensorboard, open new terminal and do cd logs --> tensorboard --logdir=.