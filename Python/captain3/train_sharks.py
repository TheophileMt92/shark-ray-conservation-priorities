#!/usr/bin/env python
"""Train a CAPTAIN 3 policy for one prioritization metric and reward setup.

    python train_sharks.py --metric IUCN --setup protpop
    python train_sharks.py --metric IUCN --setup protpop_worst [--worst-weight 1]

protpop_worst needs the calibration of the same configuration (calibrate_sharks.py).
Results go to <SHARK_RUNS_DIR>/models/<run name> (default name <metric>_<setup>):
settings.json, training_log.tsv, trained_weights.npy (last epoch) and
weights_history/ with the weights of the last --keep-weights epochs, which
predict_sharks.py uses as replicates (as CAPTAIN 2's sample_weights_from_last_iter).

Memory: one environment with all 1,000 species takes ~1.5-2 GB. On CPU, each worker
process holds its own copy (plus one in the main process), so keep --workers low on a
laptop; on a GPU a single runner evaluates the perturbations in sequence.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import time

import numpy as np
import shark_common as sc

import captain as cn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sc.add_config_arguments(parser)
    parser.add_argument(
        "--worst-weight",
        type=float,
        default=1.0,
        help="protpop_worst: weight of the (calibrated) worst-served term, relative to protpop",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--perturbations", type=int, default=30, help="ES perturbations evaluated per epoch"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU worker processes (each holds a copy of the environment); ignored on GPU",
    )
    parser.add_argument("--learning-rate", type=float, default=sc.LEARNING_RATE)
    parser.add_argument("--sigma", type=float, default=sc.INITIAL_SIGMA)
    parser.add_argument("--advantage-norm", default="zscore", choices=["zscore", "rank", "none"])
    parser.add_argument(
        "--keep-weights",
        type=int,
        default=50,
        help="keep the weights of the last N epochs (prediction replicates)",
    )
    parser.add_argument(
        "--plot-freq", type=int, default=50, help="map of the protected cells every N epochs"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or sc.default_device()
    run_name = args.run_name or sc.default_run_name(args)
    run_dir = sc.MODELS_DIR / run_name
    if (run_dir / "settings.json").exists():
        if not args.overwrite:
            raise FileExistsError(f"{run_dir} exists: use --overwrite or another --run-name")
        shutil.rmtree(run_dir)
    history_dir = run_dir / "weights_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    calibration = None
    if args.setup == "protpop_worst":
        calib_file = sc.calibration_file(args)
        if not calib_file.exists():
            raise FileNotFoundError(
                f"{calib_file} not found: run calibrate_sharks.py with the same "
                f"--metric/--setup/--n-decisions/--max-species first"
            )
        with open(calib_file) as f:
            calibration = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    data = sc.load_prepared(args.max_species, args.subset_seed)
    n_runners = 1 if device == "cuda" else max(1, args.workers)
    print(f"Creating {n_runners} episode runner(s) on {device}...")
    runners = [sc.create_episode_runner(args, data) for _ in range(n_runners)]
    del data
    episode = runners[0]
    print(sc.describe_runner(episode))

    trainer = cn.EvolStrategiesTrainer(
        runners,
        initial_coeffs=episode.policy.get_flat_weights(),
        scheduler=cn.LearningScheduler(initial_alpha=args.learning_rate, initial_sigma=args.sigma),
        epsilon_reward=0.5,
        n_perturbations=args.perturbations,
        seed=args.seed,
        advantage_normalization=None if args.advantage_norm == "none" else args.advantage_norm,
    )
    if calibration is not None:
        trainer.load_reward_calibration(calib_file)
        print(f"Rewards calibrated with {calib_file.name}: {calibration}")

    config = sc.config_settings(args)
    settings = {
        "run_name": run_name,
        "config": config,
        "config_hash": sc.config_hash(config),
        "reward_weights": {
            "protpop": 1.0,
            **({"worst": args.worst_weight} if args.setup == "protpop_worst" else {}),
        },
        "calibration": (
            {"file": calib_file.name, "multipliers": calibration}
            if calibration is not None
            else None
        ),
        "training": {
            "epochs": args.epochs,
            "perturbations": args.perturbations,
            "learning_rate": args.learning_rate,
            "sigma": args.sigma,
            "advantage_normalization": args.advantage_norm,
            "seed": args.seed,
            "keep_weights": args.keep_weights,
            "device": device,
            "workers": n_runners,
        },
        "captain_commit": sc.captain_commit(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    logger = cn.algorithms.TrainingLogger(
        trainer=trainer,
        episode=episode,
        results_dir=run_dir,
        log_file="training_log.tsv",
        weights_file="trained_weights.npy",
        plot_freq=args.plot_freq,
        figsize=(10, 5),
    )

    print(f"\nTraining {run_name} for {args.epochs} epochs...")
    t_start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        avg_reward, summary = trainer.train_epoch()
        logger.log_epoch(epoch, avg_reward, summary, time.time() - t0)
        np.save(history_dir / f"epoch_{epoch:04d}.npy", trainer.get_weights())
        old = epoch - args.keep_weights
        if old >= 0:
            (history_dir / f"epoch_{old:04d}.npy").unlink(missing_ok=True)

    settings["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    settings["epochs_completed"] = args.epochs
    settings["final_running_reward"] = float(trainer.running_reward)
    with open(run_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Training complete in {time.time() - t_start:.0f} s, results in {run_dir}")

    cn.plots.plot_rl_rewards(
        logger.log_path,
        title=f"Reward over training ({run_name})",
        outfile=run_dir / "reward_over_training.png",
        reward_col="avg_reward",
    )
    trainer.close()


if __name__ == "__main__":
    main()
