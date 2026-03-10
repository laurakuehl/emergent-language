import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze emergent language from evaluate.py message logs")
    parser.add_argument("--messages-jsonl", required=True, help="Path to message records JSONL")
    parser.add_argument("--summary-json", type=str, help="Optional output path for metrics JSON")
    parser.add_argument("--max-pairs", type=int, default=20000, help="Maximum record pairs for topographic similarity")
    return parser.parse_args()


def load_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def entropy_from_counter(counter):
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=np.float64)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def mutual_information(x, y):
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 0.0
    px = Counter(x)
    py = Counter(y)
    pxy = Counter(zip(x, y))
    n = float(len(x))
    mi = 0.0
    for (xi, yi), c in pxy.items():
        p_xy = c / n
        p_x = px[xi] / n
        p_y = py[yi] / n
        mi += p_xy * np.log2((p_xy / (p_x * p_y)) + 1e-12)
    return float(mi)


def rankdata(values):
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank
        i = j
    return np.array(ranks, dtype=np.float64)


def spearman_correlation(x, y):
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    rx = rankdata(x)
    ry = rankdata(y)
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def hamming_distance(a, b):
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    return float(sum(1 for ai, bi in zip(a, b) if ai != bi)) / float(len(a))


def build_sequences(records):
    seqs = {}
    meanings = {}
    for r in records:
        key = (
            r["channel_mode"],
            r["episode_id"],
            r["episode_seed"],
            r["batch_index"],
            r["agent_index"]
        )
        if key not in seqs:
            seqs[key] = []
            meanings[key] = (
                int(r["target_color"]),
                int(r["target_shape"]),
                int(r["goal_agent_id"])
            )
        seqs[key].append((int(r["timestep"]), int(r["token"])))

    ordered = []
    for key, entries in seqs.items():
        entries.sort(key=lambda x: x[0])
        tokens = tuple(tok for _, tok in entries)
        ordered.append((key, meanings[key], tokens))
    return ordered


def topographic_similarity(records, max_pairs):
    seq_items = build_sequences(records)
    if len(seq_items) < 2:
        return 0.0

    pair_iter = list(combinations(range(len(seq_items)), 2))
    if len(pair_iter) > max_pairs:
        rng = np.random.RandomState(0)
        chosen = rng.choice(len(pair_iter), size=max_pairs, replace=False)
        pair_iter = [pair_iter[i] for i in chosen]

    meaning_dists = []
    message_dists = []
    for i, j in pair_iter:
        _, meaning_i, msg_i = seq_items[i]
        _, meaning_j, msg_j = seq_items[j]
        meaning_dists.append(hamming_distance(meaning_i, meaning_j))
        message_dists.append(hamming_distance(msg_i, msg_j))

    return spearman_correlation(meaning_dists, message_dists)


def meaning_token_purity(records):
    if not records:
        return 0.0
    by_meaning = defaultdict(Counter)
    for r in records:
        meaning = (int(r["target_color"]), int(r["target_shape"]), int(r["goal_agent_id"]))
        by_meaning[meaning][int(r["token"])] += 1
    dominant = sum(max(counter.values()) for counter in by_meaning.values())
    total = sum(sum(counter.values()) for counter in by_meaning.values())
    return float(dominant) / float(total) if total else 0.0


def summarize_records(records, max_pairs):
    if not records:
        return {
            "n_messages": 0,
            "n_unique_tokens": 0,
            "n_unique_meanings": 0,
            "token_entropy_bits": 0.0,
            "mi_token_target_color_bits": 0.0,
            "mi_token_target_shape_bits": 0.0,
            "mi_token_goal_agent_bits": 0.0,
            "topographic_similarity": 0.0,
            "meaning_token_purity": 0.0
        }

    tokens = [int(r["token"]) for r in records]
    colors = [int(r["target_color"]) for r in records]
    shapes = [int(r["target_shape"]) for r in records]
    goal_agents = [int(r["goal_agent_id"]) for r in records]
    meanings = [(c, s, g) for c, s, g in zip(colors, shapes, goal_agents)]

    return {
        "n_messages": len(records),
        "n_unique_tokens": len(set(tokens)),
        "n_unique_meanings": len(set(meanings)),
        "token_entropy_bits": entropy_from_counter(Counter(tokens)),
        "mi_token_target_color_bits": mutual_information(tokens, colors),
        "mi_token_target_shape_bits": mutual_information(tokens, shapes),
        "mi_token_goal_agent_bits": mutual_information(tokens, goal_agents),
        "topographic_similarity": topographic_similarity(records, max_pairs=max_pairs),
        "meaning_token_purity": meaning_token_purity(records)
    }


def split_records(records):
    seen = [r for r in records if int(r.get("is_holdout_combo", 0)) == 0]
    holdout = [r for r in records if int(r.get("is_holdout_combo", 0)) == 1]
    return {"all": records, "seen": seen, "holdout": holdout}


def main():
    args = parse_args()
    records = load_records(args.messages_jsonl)
    by_mode = defaultdict(list)
    for r in records:
        by_mode[r.get("channel_mode", "unknown")].append(r)

    summary = {"modes": {}, "total_messages": len(records)}
    for mode, mode_records in sorted(by_mode.items()):
        split = split_records(mode_records)
        summary["modes"][mode] = {
            split_name: summarize_records(split_records_, max_pairs=args.max_pairs)
            for split_name, split_records_ in split.items()
        }

    for mode in sorted(summary["modes"].keys()):
        overall = summary["modes"][mode]["all"]
        print(
            "[%s] messages=%d entropy=%.4f MI(color)=%.4f topo=%.4f purity=%.4f"
            % (
                mode,
                overall["n_messages"],
                overall["token_entropy_bits"],
                overall["mi_token_target_color_bits"],
                overall["topographic_similarity"],
                overall["meaning_token_purity"]
            )
        )

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("Saved language analysis to %s" % args.summary_json)


if __name__ == "__main__":
    main()

