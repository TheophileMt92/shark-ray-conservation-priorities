#!/usr/bin/env python
"""Reward calibration for the protpop_worst setup.

Runs probe episodes with perturbations of the initial policy and derives one
multiplier per reward term so that each term varies with a standard deviation of 1
across probes (EvolStrategiesTrainer.get_reward_calibrated_weights). Training then
uses --worst-weight as the importance of the worst-served term relative to protpop.

One calibration per metric (the protpop class weights differ between metrics). The
file name holds a hash of the configuration (shark_common.config_settings), and
train_sharks.py looks it up the same way. Always computed for the single global agent
(DD-006).

    python calibrate_sharks.py --metric IUCN --setup protpop_worst [--probes 40]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import shark_common as sc

import captain as cn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sc.add_config_arguments(parser)
    parser.add_argument(
        "--probes",
        type=int,
        default=40,
        help="probe episodes (half small, half larger perturbations)",
    )
    args = parser.parse_args()
    if args.setup != "protpop_worst":
        parser.error("only the protpop_worst setup has more than one reward term to calibrate")

    out_file = sc.calibration_file(args)
    episode = sc.create_episode_runner(args)
    print(sc.describe_runner(episode))
    print(f"Calibration file: {out_file}")

    trainer = cn.EvolStrategiesTrainer(
        [episode],
        initial_coeffs=episode.policy.get_flat_weights(),
        n_perturbations=2,
        seed=args.seed,
    )
    np.random.seed(args.seed)  # probe perturbations are drawn with np.random
    t0 = time.time()
    multipliers = trainer.get_reward_calibrated_weights(n_probes=args.probes, verbose=True)
    print(f"{args.probes} probes in {time.time() - t0:.0f} s")

    calib = {name: float(m) for name, m in zip(trainer.reward_names, multipliers, strict=True)}
    config = sc.config_settings(args)
    calib.update(
        {
            "_config": config,
            "_config_hash": sc.config_hash(config),
            "_n_probes": args.probes,
            "_seed": args.seed,
            "_captain_commit": sc.captain_commit(),
            "_created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(calib, f, indent=4)

    names = trainer.reward_names
    print(
        "Multipliers (1 / std across probes): "
        + ", ".join(f"{n} x{m:.2f}" for n, m in zip(names, multipliers, strict=True))
    )
    print(f"Saved {out_file}")
    trainer.close()


if __name__ == "__main__":
    main()
