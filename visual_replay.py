import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Visual replay for evaluate.py episode logs")
    parser.add_argument("--episodes-jsonl", required=True, help="Path from evaluate.py --save-episodes")
    parser.add_argument("--episode-id", type=int, default=0, help="Episode id to replay")
    parser.add_argument("--channel-mode", type=str, default=None, help="Optional mode filter (normal/muted/shuffled/random)")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index to replay")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Optional cap on timesteps shown")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for playback/save")
    parser.add_argument("--agent-size", type=float, default=130.0, help="Marker size for agents")
    parser.add_argument("--landmark-size", type=float, default=130.0, help="Marker size for landmarks")
    parser.add_argument("--save", type=str, default=None, help="Output animation path (.gif or .mp4)")
    parser.add_argument("--dpi", type=int, default=140, help="DPI when saving animation")
    parser.add_argument("--show", action="store_true", help="Display the animation window")
    parser.add_argument("--show-trails", action="store_true", help="Draw path trails for each agent")
    parser.add_argument("--show-tokens", action="store_true", help="Show utterance token above each agent")
    parser.add_argument("--hide-goals", action="store_true", help="Hide target combo/goal overlays")
    parser.add_argument("--show-goal-lines", action="store_true", help="Draw dashed lines to uniquely matched target landmarks")
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


def build_frames(episode, batch_index, max_timesteps=None):
    timesteps = episode["timesteps"]
    if max_timesteps is not None:
        timesteps = timesteps[:max_timesteps]

    frames = [np.array(episode["initial_locations"][batch_index], dtype=float)]
    tokens = [None]
    labels = ["init"]
    for step in timesteps:
        frames.append(np.array(step["locations"][batch_index], dtype=float))
        step_tokens = step.get("utterance_tokens")
        tokens.append(step_tokens[batch_index] if step_tokens is not None else None)
        labels.append("t=%d" % int(step["timestep"]))
    return np.stack(frames, axis=0), tokens, labels


def compute_limits(frame_positions):
    all_pos = frame_positions.reshape(-1, 2)
    x_min, y_min = all_pos.min(axis=0)
    x_max, y_max = all_pos.max(axis=0)

    dx = max(x_max - x_min, 1.0)
    dy = max(y_max - y_min, 1.0)
    pad_x = 0.08 * dx
    pad_y = 0.08 * dy
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


def infer_goal_info(episode, batch_index, n_agents, n_landmarks):
    target_combos = episode.get("target_combos")
    initial_physical = episode.get("initial_physical")

    agent_combos = None
    if target_combos is not None and batch_index < len(target_combos):
        agent_combos = target_combos[batch_index]

    landmark_attrs = None
    if initial_physical is not None and batch_index < len(initial_physical):
        attrs = np.array(initial_physical[batch_index], dtype=float)
        landmark_attrs = [
            (int(round(v[0])), int(round(v[1])))
            for v in attrs[n_agents : n_agents + n_landmarks]
        ]

    goals = []
    for a in range(n_agents):
        combo = None
        if agent_combos is not None and a < len(agent_combos):
            c = agent_combos[a]
            combo = (int(c[0]), int(c[1]))

        matches = []
        if combo is not None and landmark_attrs is not None:
            for lm, lm_attr in enumerate(landmark_attrs):
                if lm_attr == combo:
                    matches.append(lm)

        goals.append(
            {
                "agent": a,
                "combo": combo,
                "matches": matches,
                "unique_landmark": matches[0] if len(matches) == 1 else None,
            }
        )

    return goals, landmark_attrs


def main():
    args = parse_args()
    episodes = load_episodes(args.episodes_jsonl)
    episode = choose_episode(episodes, args.episode_id, args.channel_mode)
    if episode is None:
        raise RuntimeError("No episode found for episode_id=%s channel_mode=%s" % (args.episode_id, args.channel_mode))

    n_agents = int(episode["num_agents"])
    n_landmarks = int(episode["num_landmarks"])
    n_entities = n_agents + n_landmarks

    frame_positions, frame_tokens, frame_labels = build_frames(episode, args.batch_index, args.max_timesteps)
    if frame_positions.shape[1] != n_entities:
        raise RuntimeError("Entity count mismatch in log: expected %d, got %d" % (n_entities, frame_positions.shape[1]))

    (x_lo, x_hi), (y_lo, y_hi) = compute_limits(frame_positions)
    agent_colors = plt.get_cmap("tab10")(np.arange(n_agents) % 10)
    landmark_colors = plt.get_cmap("Set2")(np.arange(n_landmarks) % 8)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    init_pos = frame_positions[0]
    init_agent_pos = init_pos[:n_agents]
    init_landmark_pos = init_pos[n_agents:]
    agent_scatter = ax.scatter(
        init_agent_pos[:, 0],
        init_agent_pos[:, 1],
        s=args.agent_size,
        c=agent_colors,
        marker="o",
        edgecolors="black",
        linewidths=0.6,
    )
    landmark_scatter = ax.scatter(
        init_landmark_pos[:, 0],
        init_landmark_pos[:, 1],
        s=args.landmark_size,
        c=landmark_colors,
        marker="s",
        edgecolors="black",
        linewidths=0.6,
    )

    agent_labels = [ax.text(0.0, 0.0, "", fontsize=9, weight="bold") for _ in range(n_agents)]
    landmark_labels = [ax.text(0.0, 0.0, "", fontsize=9) for _ in range(n_landmarks)]
    token_labels = [ax.text(0.0, 0.0, "", fontsize=8, color="black") for _ in range(n_agents)]
    trail_lines = [ax.plot([], [], color=agent_colors[i], alpha=0.55, lw=1.4)[0] for i in range(n_agents)]
    goal_lines = [None for _ in range(n_agents)]

    goals, landmark_attrs = infer_goal_info(episode, args.batch_index, n_agents, n_landmarks)
    if not args.hide_goals:
        goal_text_lines = ["Goals (A -> [color, shape])"]
        for g in goals:
            combo_txt = str(list(g["combo"])) if g["combo"] is not None else "unknown"
            if len(g["matches"]) == 0:
                match_txt = "no matching landmark"
            elif len(g["matches"]) == 1:
                match_txt = "L%d" % g["matches"][0]
            else:
                match_txt = "ambiguous: " + ",".join("L%d" % m for m in g["matches"])
            goal_text_lines.append("A%d -> %s (%s)" % (g["agent"], combo_txt, match_txt))

        if landmark_attrs is not None:
            goal_text_lines.append("")
            goal_text_lines.append("Landmarks [color, shape]")
            for lm, attr in enumerate(landmark_attrs):
                goal_text_lines.append("L%d -> %s" % (lm, str(list(attr))))

        ax.text(
            1.02,
            1.0,
            "\n".join(goal_text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        for i, g in enumerate(goals):
            for lm in g["matches"]:
                lx, ly = init_landmark_pos[lm]
                ax.scatter(
                    [lx],
                    [ly],
                    s=args.landmark_size * (1.9 if g["unique_landmark"] == lm else 1.45),
                    facecolors="none",
                    edgecolors=[agent_colors[i]],
                    linewidths=1.8 if g["unique_landmark"] == lm else 1.0,
                    marker="o",
                    alpha=0.95 if g["unique_landmark"] == lm else 0.5,
                )
            if args.show_goal_lines and g["unique_landmark"] is not None:
                goal_lines[i] = ax.plot([], [], color=agent_colors[i], ls="--", lw=1.2, alpha=0.6)[0]

    title = ax.set_title("")

    def draw_frame(frame_idx):
        pos = frame_positions[frame_idx]
        agent_pos = pos[:n_agents]
        landmark_pos = pos[n_agents:]
        agent_scatter.set_offsets(agent_pos)
        landmark_scatter.set_offsets(landmark_pos)

        for i in range(n_agents):
            x, y = agent_pos[i]
            agent_labels[i].set_position((x + 0.08, y + 0.08))
            agent_labels[i].set_text("A%d" % i)

            if args.show_tokens and frame_tokens[frame_idx] is not None:
                token_labels[i].set_position((x + 0.08, y - 0.22))
                token_labels[i].set_text("tok=%s" % str(frame_tokens[frame_idx][i]))
            else:
                token_labels[i].set_text("")

            if args.show_trails:
                hist = frame_positions[: frame_idx + 1, i]
                trail_lines[i].set_data(hist[:, 0], hist[:, 1])
            else:
                trail_lines[i].set_data([], [])

            if goal_lines[i] is not None:
                lm = goals[i]["unique_landmark"]
                tx, ty = landmark_pos[lm]
                goal_lines[i].set_data([x, tx], [y, ty])

        for j in range(n_landmarks):
            x, y = landmark_pos[j]
            landmark_labels[j].set_position((x + 0.08, y + 0.08))
            landmark_labels[j].set_text("L%d" % j)

        title.set_text(
            "Episode %d | mode=%s | %s | agents=%d landmarks=%d"
            % (
                int(episode["episode_id"]),
                str(episode.get("channel_mode")),
                frame_labels[frame_idx],
                n_agents,
                n_landmarks,
            )
        )

    def init():
        draw_frame(0)
        return []

    interval_ms = int(1000 / max(1, args.fps))
    anim = FuncAnimation(fig, draw_frame, init_func=init, frames=frame_positions.shape[0], interval=interval_ms, blit=False, repeat=False)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix == ".gif":
            anim.save(str(out_path), writer=PillowWriter(fps=args.fps), dpi=args.dpi)
        elif suffix == ".mp4":
            anim.save(str(out_path), writer="ffmpeg", dpi=args.dpi, fps=args.fps)
        else:
            raise RuntimeError("Unsupported output format '%s'. Use .gif or .mp4" % suffix)
        print("Saved animation to %s" % out_path)

    if args.show or not args.save:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
