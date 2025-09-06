from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from setup_env import StreetFighter
from model import LOG_DIR, OPT_DIR
import optuna
import warnings
import zipfile
import pickle

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
warnings.filterwarnings("ignore", category=FutureWarning, module="stable_baselines3")

def extract_hyperparameters_from_zip(model_path):
    """Try to extract hyperparameters from the model zip file"""
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            with zip_ref.open('data') as f:
                # Try different ways to load the data
                try:
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
    except:
        return None

print("🎯 Smart loading of Kaggle model...")

# Creating environment
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model_path = './opt/trial_3_best_model.zip'

# Strategy 1: Try to extract hyperparameters from the model file itself
print("🔍 Attempting to extract hyperparameters from model file...")
model_params = extract_hyperparameters_from_zip(model_path)

if model_params:
    print(f"✅ Extracted hyperparameters from model: {model_params}")
else:
    print("❌ Could not extract hyperparameters from model file")
    
    # Strategy 2: Try to load from Optuna study
    try:
        print("🔍 Attempting to load from Optuna study...")
        storage = optuna.storages.RDBStorage(
            url="sqlite:///optuna_study.db",
            engine_kwargs={"connect_args": {"timeout": 10}}
        )
        study = optuna.load_study(study_name="streetfighter", storage=storage)
        model_params = study.best_params.copy()
        model_params['n_steps'] = 2304  # Adjust as needed
        model_params['learning_rate'] = 5e-7  # Adjust as needed
        print(f"✅ Loaded from Optuna study: {model_params}")
    except Exception as e:
        print(f"❌ Could not load from Optuna: {e}")
        
        # Strategy 3: Use fallback hyperparameters
        print("🔄 Using fallback hyperparameters...")
        model_params = {
            'n_steps': 2304,
            'gamma': 0.99,
            'learning_rate': 5e-7,
            'clip_range': 0.2,
            'gae_lambda': 0.95
        }
        print(f"Using fallback: {model_params}")

# Create model with proper hyperparameters
print("🏗️ Creating model with hyperparameters...")
model = PPO('CnnPolicy', env, verbose=0, **model_params)

# Load weights
print(f"📥 Loading weights from: {model_path}")
try:
    model.set_parameters(model_path)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
    exit(1)

# Evaluate
print("🎮 Starting evaluation...")
try:
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True)
    print(f"🏆 Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
except Exception as e:
    print(f"❌ Evaluation with rendering failed: {e}")
    if "rendering" in str(e) or "classic_control" in str(e):
        print("🔄 Rendering issue - evaluating without render...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=False)
        print(f"🏆 Mean reward (no render): {mean_reward:.2f} ± {std_reward:.2f}")
        print("💡 To fix rendering: pip install 'gym[classic_control]' or use gym==0.21.0")
    else:
        raise e

env.close()
print("✅ Evaluation complete!")

# Bonus: Show model info
print("\n📊 Model Information:")
print(f"   Policy: {model.policy}")
print(f"   Environment: {model.env}")
print(f"   Observation space: {model.observation_space}")
print(f"   Action space: {model.action_space}")