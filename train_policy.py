import torch
import torch.nn as nn
import torch.optim as optim
from two_phase_env import TwoPhaseEnv

# === Define the policy network ===
class ModifierPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ModifierPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()  # Output: probabilities for each action
        )

    def forward(self, x):
        return self.fc(x)

# === Initialize environment and policy ===
env = TwoPhaseEnv(action_dim=5, max_weight=10)
policy = ModifierPolicy(obs_dim=env.observation_space.shape[0], action_dim=env.action_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

num_episodes = 500

# === Training loop ===
for episode in range(num_episodes):
    state = torch.FloatTensor(env.reset())

    # Forward pass to get action probabilities
    action_probs = policy(state)

    # Sample binary actions from Bernoulli distributions
    dist = torch.distributions.Bernoulli(action_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()

    # Apply action and observe reward
    _, reward, _, info = env.step(action.detach().numpy())

    # Compute policy gradient loss
    loss = -log_prob * reward

    # Update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 20 == 0:
        print(f"Ep {episode:03d} | Reward: {reward:.2f} | Score: {info['executor_score']:.2f} | Cost: {info['cost']}")

