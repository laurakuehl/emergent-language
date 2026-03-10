import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch

import configs
from modules.agent import AgentModule
from modules.game import GameModule


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agents with communication ablations and trajectory logging")
    parser.add_argument("--model-weights", required=True, help="Path to saved model produced by train.py")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes per channel mode")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible evaluation")
    parser.add_argument("--success-threshold", type=float, default=1.0, help="Success if all agents are within this distance")
    parser.add_argument(
        "--channel-modes",
        nargs="+",
        choices=["normal", "muted", "shuffled", "random"],
        default=None,
        help="Communication channel settings to evaluate"
    )

    parser.add_argument("--save-episodes", type=str, help="Optional JSONL output for episode-level rollouts")
    parser.add_argument("--save-messages", type=str, help="Optional JSONL output for per-message records")
    parser.add_argument("--summary-json", type=str, help="Optional JSON output for aggregate summary")

    parser.add_argument("--no-utterances", action="store_true")
    parser.add_argument("--penalize-words", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-timesteps", type=int)
    parser.add_argument("--num-shapes", type=int)
    parser.add_argument("--num-colors", type=int)
    parser.add_argument("--max-agents", type=int)
    parser.add_argument("--min-agents", type=int)
    parser.add_argument("--max-landmarks", type=int)
    parser.add_argument("--min-landmarks", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--world-dim", type=int)
    parser.add_argument("--oov-prob", type=int)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--holdout-combos", type=str, help="Comma-separated color-shape pairs, e.g. '0-1,2-0'")
    parser.add_argument("--holdout-mode", choices=["off", "exclude", "only"], default="off")

    return parser.parse_args()


def set_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agent(path, args):
    map_location = None if args.use_cuda else "cpu"
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=map_location)
    if hasattr(loaded, "reset") and hasattr(loaded, "forward"):
        agent = loaded
    else:
        agent = AgentModule(configs.get_agent_config(vars(args)))
        agent.load_state_dict(loaded)
    if args.use_cuda:
        agent.cuda()
    return agent


def build_episode_specs(game_config, n_episodes, seed):
    rng = np.random.RandomState(seed)
    specs = []
    for _ in range(n_episodes):
        num_agents = int(rng.randint(game_config.min_agents, game_config.max_agents + 1))
        num_landmarks = int(rng.randint(game_config.min_landmarks, game_config.max_landmarks + 1))
        episode_seed = int(rng.randint(0, 2**31 - 1))
        specs.append((num_agents, num_landmarks, episode_seed))
    return specs


def apply_channel_mode(utterances, mode):
    if mode == "normal":
        return utterances
    if mode == "muted":
        return torch.zeros_like(utterances)
    if mode == "shuffled":
        shuffled = utterances.clone()
        for b in range(utterances.size(0)):
            perm = torch.randperm(utterances.size(1), device=utterances.device)
            shuffled[b] = utterances[b, perm]
        return shuffled
    if mode == "random":
        random_tokens = torch.randint(
            0, utterances.size(2), (utterances.size(0), utterances.size(1)), device=utterances.device
        )
        random_utterances = torch.zeros_like(utterances)
        random_utterances.scatter_(2, random_tokens.unsqueeze(2), 1.0)
        return random_utterances
    raise ValueError("Unknown channel mode: %s" % mode)


def target_attributes(game):
    landmarks = game.locations[:, game.num_agents:, :].detach()
    targets = game.sorted_goals.detach()
    sq_dist = ((landmarks.unsqueeze(1) - targets.unsqueeze(2)) ** 2).sum(-1)
    target_landmarks = sq_dist.argmin(dim=2)
    target_entities = target_landmarks + game.num_agents
    target_colors = torch.gather(game.physical[:, :, 0].long(), 1, target_entities.long())
    target_shapes = torch.gather(game.physical[:, :, 1].long(), 1, target_entities.long())
    return target_landmarks, target_colors, target_shapes


def rollout_episode(agent, game_config, num_agents, num_landmarks, channel_mode, episode_id, episode_seed, success_threshold):
    game = GameModule(game_config, num_agents, num_landmarks)
    if game_config.use_cuda:
        game.cuda()

    agent.reset()
    agent.train(False)
    holdout_set = set(tuple(c) for c in getattr(game_config, "holdout_combos", []))
    target_landmarks, target_colors, target_shapes = target_attributes(game)
    targets = game.sorted_goals.detach()
    initial_locations = game.locations.detach().cpu().tolist()
    initial_physical = game.physical.detach().cpu().tolist()

    total_cost = 0.0
    timesteps = []
    message_records = []

    with torch.no_grad():
        for t in range(agent.time_horizon):
            movements = torch.zeros(
                game.batch_size, game.num_entities, agent.movement_dim_size, device=game.locations.device
            )
            utterances = None
            goal_predictions = None

            if agent.using_utterances:
                utterances = torch.zeros(
                    game.batch_size, game.num_agents, agent.vocab_size, device=game.locations.device
                )
                goal_predictions = torch.zeros(
                    game.batch_size, game.num_agents, game.num_agents, agent.goal_size, device=game.locations.device
                )

            for agent_idx in range(game.num_agents):
                physical_feat = agent.get_physical_feat(game, agent_idx)
                utterance_feat = agent.get_utterance_feat(game, agent_idx, goal_predictions)
                agent.get_action(game, agent_idx, physical_feat, utterance_feat, movements, utterances)

            if utterances is not None:
                utterances = apply_channel_mode(utterances, channel_mode)
                utterance_tokens = utterances.argmax(dim=2)
            else:
                utterance_tokens = None

            step_cost = game(movements, goal_predictions, utterances)
            if utterances is not None and agent.penalizing_words:
                step_cost = step_cost + agent.word_counter(utterances)
            total_cost += float(step_cost.item())

            timesteps.append({
                "timestep": t,
                "locations": game.locations.detach().cpu().tolist(),
                "movements": movements[:, :game.num_agents].detach().cpu().tolist(),
                "utterance_tokens": utterance_tokens.detach().cpu().tolist() if utterance_tokens is not None else None
            })

            if utterance_tokens is not None:
                for b in range(game.batch_size):
                    for a in range(game.num_agents):
                        color = int(target_colors[b, a].item())
                        shape = int(target_shapes[b, a].item())
                        message_records.append({
                            "episode_id": episode_id,
                            "episode_seed": episode_seed,
                            "timestep": t,
                            "batch_index": b,
                            "agent_index": a,
                            "token": int(utterance_tokens[b, a].item()),
                            "target_color": color,
                            "target_shape": shape,
                            "target_landmark": int(target_landmarks[b, a].item()),
                            "goal_agent_id": a,
                            "goal_x": float(targets[b, a, 0].item()),
                            "goal_y": float(targets[b, a, 1].item()),
                            "num_agents": game.num_agents,
                            "num_landmarks": game.num_landmarks,
                            "channel_mode": channel_mode,
                            "holdout_mode": game_config.holdout_mode,
                            "is_holdout_combo": int((color, shape) in holdout_set)
                        })

    final_distances = torch.norm(game.locations[:, :game.num_agents, :] - game.sorted_goals, dim=2)
    avg_final_distance = float(final_distances.mean().item())
    success_rate = float((final_distances <= success_threshold).all(dim=1).float().mean().item())
    per_agent_cost = total_cost / float(game.batch_size * game.num_agents)

    target_combo_grid = []
    for b in range(game.batch_size):
        per_agent = []
        for a in range(game.num_agents):
            per_agent.append([int(target_colors[b, a].item()), int(target_shapes[b, a].item())])
        target_combo_grid.append(per_agent)

    episode_summary = {
        "episode_id": episode_id,
        "episode_seed": episode_seed,
        "channel_mode": channel_mode,
        "num_agents": game.num_agents,
        "num_landmarks": game.num_landmarks,
        "avg_final_distance": avg_final_distance,
        "success_rate": success_rate,
        "per_agent_cost": per_agent_cost,
        "holdout_mode": game_config.holdout_mode,
        "holdout_combos": [list(c) for c in game_config.holdout_combos],
        "target_combos": target_combo_grid,
        "initial_locations": initial_locations,
        "initial_physical": initial_physical,
        "timesteps": timesteps
    }
    return episode_summary, message_records


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std()) if len(arr) else 0.0,
        "min": float(arr.min()) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0
    }


def main():
    args = parse_args()
    if args.use_cuda and not torch.cuda.is_available():
        raise RuntimeError("--use-cuda was passed, but CUDA is not available")

    game_config = configs.get_game_config(vars(args))
    set_seed(args.seed, args.use_cuda)
    agent = load_agent(args.model_weights, args)
    if game_config.use_utterances != agent.using_utterances:
        game_config = game_config._replace(use_utterances=agent.using_utterances)

    if args.channel_modes is None:
        channel_modes = ["normal", "muted", "shuffled", "random"] if agent.using_utterances else ["normal"]
    else:
        channel_modes = args.channel_modes

    episode_specs = build_episode_specs(game_config, args.episodes, args.seed)
    aggregate = defaultdict(lambda: defaultdict(list))
    episode_lines = []
    message_lines = []

    for mode in channel_modes:
        if not agent.using_utterances and mode != "normal":
            continue
        for episode_id, (num_agents, num_landmarks, episode_seed) in enumerate(episode_specs):
            set_seed(episode_seed, args.use_cuda)
            episode, records = rollout_episode(
                agent=agent,
                game_config=game_config,
                num_agents=num_agents,
                num_landmarks=num_landmarks,
                channel_mode=mode,
                episode_id=episode_id,
                episode_seed=episode_seed,
                success_threshold=args.success_threshold
            )
            episode_lines.append(episode)
            message_lines.extend(records)
            aggregate[mode]["avg_final_distance"].append(episode["avg_final_distance"])
            aggregate[mode]["success_rate"].append(episode["success_rate"])
            aggregate[mode]["per_agent_cost"].append(episode["per_agent_cost"])

    summary = {"seed": args.seed, "episodes_per_mode": args.episodes, "modes": {}}
    for mode in sorted(aggregate.keys()):
        summary["modes"][mode] = {
            "avg_final_distance": summarize(aggregate[mode]["avg_final_distance"]),
            "success_rate": summarize(aggregate[mode]["success_rate"]),
            "per_agent_cost": summarize(aggregate[mode]["per_agent_cost"])
        }

    for mode, metrics in summary["modes"].items():
        print(
            "[%s] success mean=%.4f std=%.4f | distance mean=%.4f | cost mean=%.4f"
            % (
                mode,
                metrics["success_rate"]["mean"],
                metrics["success_rate"]["std"],
                metrics["avg_final_distance"]["mean"],
                metrics["per_agent_cost"]["mean"]
            )
        )

    if args.save_episodes:
        with open(args.save_episodes, "w") as f:
            for line in episode_lines:
                f.write(json.dumps(line) + "\n")
        print("Saved episode logs to %s" % args.save_episodes)

    if args.save_messages:
        with open(args.save_messages, "w") as f:
            for line in message_lines:
                f.write(json.dumps(line) + "\n")
        print("Saved message logs to %s" % args.save_messages)

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("Saved summary to %s" % args.summary_json)


if __name__ == "__main__":
    main()
