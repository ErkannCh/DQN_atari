import yaml
import torch
from pathlib import Path

from dqn.utils.wrappers import make_atari_env
from dqn.models.cnn_dqn import CNNDQN


def load_config(path: str = "configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(num_episodes: int = 5, render: bool = False):
    config = load_config()

    env_name = config["env"]["name"]
    frame_skip = config["env"]["frame_skip"]
    num_stack = config["model"]["input_shape"][0]

    render_mode = "human" if render else None
    env = make_atari_env(
        env_id=env_name,
        frame_skip=frame_skip,
        num_stack=num_stack,
        render_mode=render_mode,
    )

    obs, info = env.reset()
    state_shape = obs.shape
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNDQN(input_shape=state_shape, num_actions=num_actions).to(device)

    ckpt_path = Path("runs/dqn_breakout_final.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    episode_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            state_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = int(q_values.argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        episode_rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{num_episodes} Reward: {ep_reward}")

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    env.close()


if __name__ == "__main__":
    evaluate(num_episodes=5, render=False)
