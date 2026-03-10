import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Text replay for evaluate.py episode logs")
    parser.add_argument("--episodes-jsonl", required=True, help="Path from evaluate.py --save-episodes")
    parser.add_argument("--episode-id", type=int, default=0, help="Episode id to replay")
    parser.add_argument("--channel-mode", type=str, default=None, help="Optional mode filter (normal/muted/shuffled/random)")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index to replay")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Optional cap on printed timesteps")
    return parser.parse_args()


def load_episodes(path):
    episodes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def choose_episode(episodes, episode_id, channel_mode):
    for ep in episodes:
        if ep.get("episode_id") != episode_id:
            continue
        if channel_mode is not None and ep.get("channel_mode") != channel_mode:
            continue
        return ep
    return None


def format_vec(vec):
    return "(%.2f, %.2f)" % (vec[0], vec[1])


def main():
    args = parse_args()
    episodes = load_episodes(args.episodes_jsonl)
    episode = choose_episode(episodes, args.episode_id, args.channel_mode)
    if episode is None:
        raise RuntimeError("No episode found for episode_id=%s channel_mode=%s" % (args.episode_id, args.channel_mode))

    b = args.batch_index
    print("Episode %d | mode=%s | seed=%s | agents=%d | landmarks=%d"
          % (
              episode["episode_id"],
              episode["channel_mode"],
              episode["episode_seed"],
              episode["num_agents"],
              episode["num_landmarks"]
          ))
    print("Target combos (color, shape) per agent:", episode["target_combos"][b])
    print("Initial agent positions:")
    for a in range(episode["num_agents"]):
        print("  agent %d: %s" % (a, format_vec(episode["initial_locations"][b][a])))
    print("Initial landmark positions:")
    for lm in range(episode["num_landmarks"]):
        idx = episode["num_agents"] + lm
        print("  landmark %d: %s" % (lm, format_vec(episode["initial_locations"][b][idx])))

    timesteps = episode["timesteps"]
    if args.max_timesteps is not None:
        timesteps = timesteps[:args.max_timesteps]

    for step in timesteps:
        print("\n[t=%d]" % step["timestep"])
        for a in range(episode["num_agents"]):
            pos = step["locations"][b][a]
            mov = step["movements"][b][a]
            token = None
            if step["utterance_tokens"] is not None:
                token = step["utterance_tokens"][b][a]
            print("  agent %d pos=%s move=%s token=%s" % (a, format_vec(pos), format_vec(mov), str(token)))


if __name__ == "__main__":
    main()

