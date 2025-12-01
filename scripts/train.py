import yaml
import torch
from collections import deque
from pathlib import Path
import time

from dqn.utils.wrappers import make_atari_env
from dqn.memory.replay_buffer import ReplayBuffer
from dqn.models.cnn_dqn import CNNDQN
from dqn.agents import DQNAgent, EpsilonConfig


def load_config(path: str = "configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
        config = load_config()
        print("Config loaded:")
        print(config)

        env_name = config["env"]["name"]
        frame_skip = config["env"]["frame_skip"]
        num_stack = config["model"]["input_shape"][0]

        training_cfg = config["training"]
        total_frames = int(training_cfg["total_frames"])
        batch_size = int(training_cfg["batch_size"])
        gamma = float(training_cfg["gamma"])
        learning_rate = float(training_cfg["learning_rate"])
        replay_buffer_size = int(training_cfg["replay_buffer_size"])

        learning_starts = int(training_cfg.get("learning_starts", 10000))
        train_freq = int(training_cfg.get("train_freq", 1))
        print_freq = int(training_cfg.get("print_freq", 10000))

        eps_cfg = EpsilonConfig(
            start=float(config["epsilon"]["start"]),
            end=float(config["epsilon"]["end"]),
            decay_frames=int(config["epsilon"]["decay_frames"]),
        )

        env = make_atari_env(
            env_id=env_name,
            frame_skip=frame_skip,
            num_stack=num_stack,
            render_mode=None,
        )

        obs, info = env.reset()
        state_shape = obs.shape

        replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            state_shape=state_shape,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = env.action_space.n

        model = CNNDQN(input_shape=state_shape, num_actions=num_actions).to(device)

        agent = DQNAgent(
            model=model,
            num_actions=num_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_config=eps_cfg,
            device=device,
        )

        frame_idx = 0
        episode_idx = 0
        episode_reward = 0.0
        episode_rewards = []
        recent_rewards = deque(maxlen=10)
        recent_losses = deque(maxlen=100)

        start_time = time.time()

        while frame_idx < total_frames:
            action = agent.select_action(obs, frame_idx)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )

            obs = next_obs
            episode_reward += reward
            frame_idx += 1

            loss_value = None
            if frame_idx >= learning_starts and frame_idx % train_freq == 0:
                loss_value = agent.update(replay_buffer, batch_size=batch_size)
                if loss_value is not None:
                    recent_losses.append(loss_value)

            if done:
                episode_idx += 1
                episode_rewards.append(episode_reward)
                recent_rewards.append(episode_reward)
                eps = agent.epsilon_by_frame(frame_idx)
                avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                elapsed = time.time() - start_time

                print(
                    f"Frame: {frame_idx}/{total_frames} | "
                    f"Episode: {episode_idx} | "
                    f"Eps: {eps:.3f} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"AvgReward(10): {avg_reward:.2f} | "
                    f"AvgLoss(100): {avg_loss:.4f} | "
                    f"Elapsed: {elapsed/60:.1f} min"
                )

                episode_reward = 0.0
                obs, info = env.reset()

            if frame_idx % print_freq == 0 and frame_idx > 0:
                eps = agent.epsilon_by_frame(frame_idx)
                avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                elapsed = time.time() - start_time
                print(
                    f"[Progress] Frame: {frame_idx}/{total_frames} | "
                    f"Eps: {eps:.3f} | "
                    f"AvgReward(10): {avg_reward:.2f} | "
                    f"AvgLoss(100): {avg_loss:.4f} | "
                    f"Elapsed: {elapsed/60:.1f} min"
                )

        env.close()

        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        save_path = runs_dir / "dqn_breakout_final.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Training finished. Model saved to: {save_path}")


if __name__ == "__main__":
    main()
