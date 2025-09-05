# Street Fighter II RL Agent

A reinforcement learning agent that learns to play Street Fighter II using PPO (Proximal Policy Optimization) with Optuna hyperparameter optimization.

## 🎮 Project Overview

This project creates an AI agent that can play Street Fighter II by:
- **Preprocessing**: Grayscaling, resizing to 84x84, and frame delta calculation
- **Action Filtering**: Using only viable attack combinations instead of all 4096 possible actions
- **Reward Function**: Based on in-game score changes
- **Training**: PPO algorithm with Optuna hyperparameter optimization

## 🛠️ Environment Setup

**CRITICAL**: This project requires **Python 3.8 EXACTLY** - no upgrades or downgrades!

### Prerequisites
1. Street Fighter II ROM file
2. CUDA-enabled GPU (recommended for training)

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/streetfighter-rl-agent.git
cd streetfighter-rl-agent

# Install dependencies
pip install -r requirements.txt

# Import ROM (place ROM in project directory first)
python -m retro.import .
```

## 📁 Project Structure

```
streetfighter-rl-agent/
├── setup_env.py          # Custom Street Fighter environment
├── train_optuna.py       # Optuna hyperparameter optimization
├── test_agent.py         # Test the trained agent
├── requirements.txt      # Project dependencies
├── logs/                 # Training logs (created during training)
├── opt/                  # Optimized models (created during training)
└── README.md
```

## 🚀 Usage

### Testing the Environment
```bash
python test_agent.py
```

### Training with Optuna Optimization
```bash
python train_optuna.py
```

### Key Features
- **Frame Preprocessing**: 84x84 grayscale with frame delta
- **Action Space**: Filtered to viable combinations only
- **Reward System**: Based on score changes
- **Hyperparameter Tuning**: Automated with Optuna
- **Frame Stacking**: 4 frames for temporal understanding

## 🎯 Training Parameters

The model optimizes these hyperparameters:
- `n_steps`: 2048-8192 (frames per training batch)
- `gamma`: 0.8-0.9999 (discount factor)
- `learning_rate`: 1e-5 to 1e-4
- `clip_range`: 0.1-0.4 (PPO clipping)
- `gae_lambda`: 0.8-0.99 (GAE smoothing)

## 📊 Training Progress

- **Quick Test**: 30,000 timesteps
- **Full Training**: 100,000+ timesteps (recommended)
- **Evaluation**: 5 episodes per trial
- **Optimization**: 10-100 trials (configurable)

## 🏆 Results

Training results and model performance will be logged to:
- `./logs/` - TensorBoard logs
- `./opt/` - Best model checkpoints

## 🔧 Hardware Requirements

- **CPU**: Multi-core recommended
- **RAM**: 8GB+ recommended
- **GPU**: CUDA-enabled GPU highly recommended for training
- **Storage**: 2GB+ for logs and models

## 📝 Notes

- Ensure ROM is properly imported before training
- Training can take 20+ hours for full optimization
- Use Kaggle/Colab for GPU acceleration if needed
- Monitor training progress via TensorBoard

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!
