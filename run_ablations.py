import argparse
import json
import os
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Run training/eval/language ablations for emergent-language")
    parser.add_argument("--output-dir", default="ablation_runs", help="Directory for all outputs")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds per variant")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--distance-threshold", type=float, default=2.0, help="For sample-efficiency epoch-to-threshold")
    parser.add_argument("--holdout-combos", type=str, help="Optional holdout split, e.g. '0-1,2-0'")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--train-extra", type=str, default="", help="Extra args appended to every train.py call")
    parser.add_argument("--eval-extra", type=str, default="", help="Extra args appended to every evaluate.py call")
    parser.add_argument("--variants", nargs="+", help="Optional subset of variants to run")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have outputs")
    return parser.parse_args()


def run_cmd(cmd):
    print("+ %s" % shlex.join(cmd))
    subprocess.run(cmd, check=True)


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def epoch_to_threshold(metrics_jsonl, threshold):
    if not os.path.exists(metrics_jsonl):
        return None
    with open(metrics_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row["avg_dist"]) <= threshold:
                return int(row["epoch"])
    return None


def variant_specs(args):
    variants = [
        ("with_comm", []),
        ("no_comm", ["--no-utterances"]),
        ("with_word_penalty", ["--penalize-words"]),
        ("small_vocab", ["--vocab-size", "8"]),
    ]
    if args.holdout_combos:
        variants.append(
            ("holdout_train_seen", ["--holdout-combos", args.holdout_combos, "--holdout-mode", "exclude"])
        )
    if args.variants:
        requested = set(args.variants)
        available = set(name for name, _ in variants)
        unknown = requested - available
        if unknown:
            raise ValueError("Unknown variants requested: %s" % ", ".join(sorted(unknown)))
        variants = [v for v in variants if v[0] in requested]
    return variants


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_extra = shlex.split(args.train_extra)
    eval_extra = shlex.split(args.eval_extra)
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    results_by_key = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing_rows = json.load(f)
            for row in existing_rows:
                results_by_key[(row["variant"], int(row["seed"]))] = row
        except Exception:
            pass

    for variant_name, variant_args in variant_specs(args):
        for seed in args.seeds:
            run_dir = os.path.join(args.output_dir, variant_name, "seed_%d" % seed)
            os.makedirs(run_dir, exist_ok=True)
            model_path = os.path.join(run_dir, "model_weights.pt")
            metrics_path = os.path.join(run_dir, "train_metrics.jsonl")
            eval_summary_path = os.path.join(run_dir, "eval_summary.json")
            episodes_path = os.path.join(run_dir, "episodes.jsonl")
            messages_path = os.path.join(run_dir, "messages.jsonl")
            lang_summary_path = os.path.join(run_dir, "language_summary.json")
            run_key = (variant_name, seed)

            if (
                args.skip_existing
                and os.path.exists(model_path)
                and os.path.exists(eval_summary_path)
                and os.path.exists(lang_summary_path)
            ):
                run_result = {
                    "variant": variant_name,
                    "seed": seed,
                    "model_path": model_path,
                    "epoch_to_distance_threshold": epoch_to_threshold(metrics_path, args.distance_threshold),
                    "eval_summary": read_json(eval_summary_path),
                    "language_summary": read_json(lang_summary_path),
                }
                if variant_name == "holdout_train_seen":
                    unseen_eval_summary = os.path.join(run_dir, "eval_unseen_summary.json")
                    unseen_lang = os.path.join(run_dir, "language_unseen_summary.json")
                    if os.path.exists(unseen_eval_summary):
                        run_result["unseen_eval_summary"] = read_json(unseen_eval_summary)
                    if os.path.exists(unseen_lang):
                        run_result["unseen_language_summary"] = read_json(unseen_lang)
                results_by_key[run_key] = run_result
                print("Skipping existing run %s seed %d" % (variant_name, seed))
                continue
            else:
                # Ensure retries don't append onto partial outputs from failed runs.
                remove_if_exists(model_path)
                remove_if_exists(metrics_path)
                remove_if_exists(eval_summary_path)
                remove_if_exists(episodes_path)
                remove_if_exists(messages_path)
                remove_if_exists(lang_summary_path)
                remove_if_exists(os.path.join(run_dir, "eval_unseen_summary.json"))
                remove_if_exists(os.path.join(run_dir, "episodes_unseen.jsonl"))
                remove_if_exists(os.path.join(run_dir, "messages_unseen.jsonl"))
                remove_if_exists(os.path.join(run_dir, "language_unseen_summary.json"))

            train_cmd = [
                args.python_bin,
                "train.py",
                "--n-epochs", str(args.n_epochs),
                "--seed", str(seed),
                "--save-model-weights", model_path,
                "--metrics-jsonl", metrics_path,
            ] + variant_args + train_extra
            if args.use_cuda:
                train_cmd.append("--use-cuda")
            run_cmd(train_cmd)

            eval_cmd = [
                args.python_bin,
                "evaluate.py",
                "--model-weights", model_path,
                "--episodes", str(args.eval_episodes),
                "--seed", str(seed),
                "--success-threshold", str(args.success_threshold),
                "--save-episodes", episodes_path,
                "--save-messages", messages_path,
                "--summary-json", eval_summary_path,
            ] + variant_args + eval_extra
            if args.use_cuda:
                eval_cmd.append("--use-cuda")
            run_cmd(eval_cmd)

            lang_cmd = [
                args.python_bin,
                "language_analysis.py",
                "--messages-jsonl", messages_path,
                "--summary-json", lang_summary_path,
            ]
            run_cmd(lang_cmd)

            run_result = {
                "variant": variant_name,
                "seed": seed,
                "model_path": model_path,
                "epoch_to_distance_threshold": epoch_to_threshold(metrics_path, args.distance_threshold),
                "eval_summary": read_json(eval_summary_path),
                "language_summary": read_json(lang_summary_path),
            }

            if variant_name == "holdout_train_seen":
                unseen_eval_summary = os.path.join(run_dir, "eval_unseen_summary.json")
                unseen_episodes = os.path.join(run_dir, "episodes_unseen.jsonl")
                unseen_messages = os.path.join(run_dir, "messages_unseen.jsonl")
                unseen_lang = os.path.join(run_dir, "language_unseen_summary.json")

                unseen_cmd = [
                    args.python_bin,
                    "evaluate.py",
                    "--model-weights", model_path,
                    "--episodes", str(args.eval_episodes),
                    "--seed", str(seed),
                    "--success-threshold", str(args.success_threshold),
                    "--holdout-combos", args.holdout_combos,
                    "--holdout-mode", "only",
                    "--save-episodes", unseen_episodes,
                    "--save-messages", unseen_messages,
                    "--summary-json", unseen_eval_summary,
                ] + eval_extra
                if args.use_cuda:
                    unseen_cmd.append("--use-cuda")
                run_cmd(unseen_cmd)

                unseen_lang_cmd = [
                    args.python_bin,
                    "language_analysis.py",
                    "--messages-jsonl", unseen_messages,
                    "--summary-json", unseen_lang,
                ]
                run_cmd(unseen_lang_cmd)
                run_result["unseen_eval_summary"] = read_json(unseen_eval_summary)
                run_result["unseen_language_summary"] = read_json(unseen_lang)

            results_by_key[run_key] = run_result

    all_results = [results_by_key[k] for k in sorted(results_by_key.keys())]
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    print("Saved ablation summary to %s" % summary_path)


if __name__ == "__main__":
    main()
