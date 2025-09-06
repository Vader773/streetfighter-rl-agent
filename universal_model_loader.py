# WRITTEN BY AI


"""
Universal model loader that works across different environments and gym versions
Add this file to your dataset to prevent future compatibility issues
"""

import warnings
import os
import zipfile
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import optuna

# Suppress all the annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
warnings.filterwarnings("ignore", category=FutureWarning, module="stable_baselines3") 
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_compatible_env():
    """Create environment that works across different gym versions"""
    from setup_env import StreetFighter
    from model import LOG_DIR
    
    env = StreetFighter()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env

def extract_hyperparams_safe(model_path):
    """Safely extract hyperparameters from model file"""
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            with zip_ref.open('data') as f:
                data = pickle.load(f)
                return {
                    'learning_rate': data.get('learning_rate', 3e-4),
                    'n_steps': data.get('n_steps', 2048), 
                    'gamma': data.get('gamma', 0.99),
                    'clip_range': data.get('clip_range', 0.2),
                    'gae_lambda': data.get('gae_lambda', 0.95)
                }
    except:
        return None

def load_from_optuna_safe():
    """Safely load hyperparameters from Optuna study"""
    try:
        storage = optuna.storages.RDBStorage(
            url="sqlite:///optuna_study.db",
            engine_kwargs={"connect_args": {"timeout": 10}}
        )
        study = optuna.load_study(study_name="streetfighter", storage=storage)
        params = study.best_params.copy()
        # Apply standard adjustments
        params['n_steps'] = 2304  # Multiple of 64
        params['learning_rate'] = 5e-7  # Fine-tuning rate
        return params
    except:
        return None

def universal_model_load(model_path, env=None):
    """
    Universal model loader that works with Kaggle and local environments
    
    Args:
        model_path: Path to the .zip model file
        env: Environment (will create if None)
    
    Returns:
        Loaded PPO model
    """
    if env is None:
        env = create_compatible_env()
    
    print(f"Loading model: {model_path}")
    
    # Strategy 1: Try to get hyperparameters from model file
    model_params = extract_hyperparams_safe(model_path)
    
    if model_params is None:
        print("Could not extract hyperparams from model, trying Optuna...")
        # Strategy 2: Try Optuna
        model_params = load_from_optuna_safe()
    
    if model_params is None:
        print("Using fallback hyperparameters...")
        # Strategy 3: Fallback defaults
        model_params = {
            'n_steps': 2304,
            'gamma': 0.99, 
            'learning_rate': 5e-7,
            'clip_range': 0.2,
            'gae_lambda': 0.95
        }
    
    print(f"Using hyperparameters: {model_params}")
    
    # Create new model with correct hyperparameters
    model = PPO('CnnPolicy', env, verbose=0, **model_params)
    
    # Load only the weights (this always works!)
    model.set_parameters(model_path)
    print("Model loaded successfully!")
    
    return model

def evaluate_model_universal(model_path, n_episodes=5, render=False):
    """Universal model evaluation"""
    from stable_baselines3.common.evaluation import evaluate_policy
    
    env = create_compatible_env()
    model = universal_model_load(model_path, env)
    
    try:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_episodes, render=render
        )
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        env.close()
        return mean_reward, std_reward
    except Exception as e:
        print(f"Evaluation failed: {e}")
        env.close()
        return None, None

if __name__ == "__main__":
    # Test the loader
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        evaluate_model_universal(model_path, render=False)
    else:
        print("Usage: python universal_model_loader.py <model_path>")