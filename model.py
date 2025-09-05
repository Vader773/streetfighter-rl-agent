import optuna # optimization import which optimizes hyperparams and trains at the same time
from stable_baselines3 import PPO # Importing Proximal Policy Optimization (PPO) algorithm for our model
from stable_baselines3.common.evaluation import evaluate_policy # This helps us evaluate our model during hyperparam tuning to find the best one
from stable_baselines3.common.monitor import Monitor # This helps us monitor our training and log results. Monitor is a great way to get logging mean episode/reward values from wrapped/multiple parallel streams.
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # This helps us stack 4 frames together to give the ai a sense of time or trajectory.
from setup_env import StreetFighter
import os
import traceback


LOG_DIR = './logs/'
OPT_DIR = './opt/'

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
        model.learn(total_timesteps=30000) # For a quick test run
        # model.learn(total_timestamps=100000) # How much to train the model....100k is optimum but takes a shit ton of time so gotta migrate to kaggle for that ig

        #---------------EVALUATING THE MODEL-------------------

        # Here, "mean)reward, _" means we are UNPACKING the results we got from our policy before passing it to evaluate function
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)  # Evaluation of our model, through our environment, for n number of GAMES to play
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number)) # Saved to our path OPT_DIR, and then formatted name such that trial number is put in place of {}
        model.save(SAVE_PATH) # IMP: Saves our model

        return mean_reward

    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed due to: {e}")
        traceback.print_exc()
        return -1000 # give placeholder value so that entire loop doenst break in case of one or two errors.


# Creating the experiment (?)
study = optuna.create_study(direction='maximize') # It is important to set direction to MAXIMIZE if we want to return positive mean rewards, if not then we must return negative mean rewards (why?)

# Running the study through our optimize agent function, where n_trials means HOW MANY DIFFERENT SETS of hyperparams we will be testing using optuna, and n_jobs specifies how many PARALLEL ENVIRONMENTS are used to train at the same time
study.optimize(optimize_agent, n_trials=10, n_jobs=1) 
# study.optimize(optimize_agent, n_trials=100, n_jobs=1) #For the actual hardcore training....estimated to take more than 22 hours!


# TODO : REINSTALL cuda with gpu this time for gods sake and remember its a 2.7gb install!!
# We gotta finish this by tomorrow (9/5/25)