#!/usr/bin/env python
"""Protection plans from a trained CAPTAIN 3 shark model.

Runs one episode per saved weight snapshot (the last --replicates training epochs, as
CAPTAIN 2's sample_weights_from_last_iter) and writes to
<SHARK_RUNS_DIR>/models/<run name>/predictions:

    replicate_<i>.npz   per replicate, in the CAPTAIN 2 output format:
                        protection_matrix (323 x 720 grid, 1 = protected),
                        protected_range_fraction (per species, file-name order), plus
                        decision (grid, 1..n = decision that protected the cell, 0 = not),
                        protected_pop_fraction, species, rewards (raw, per term)
    CAPTAIN3_<metric>_<setup>_averaged_budget0.1_replicates<n>.csv
                        PUID (1..232560, row-major grid cells) and Priority (fraction of
                        replicates protecting the cell): same columns as the CAPTAIN 2
                        .rds files used in R
    protection_frequency.tif, mean_decision.tif
                        Priority and mean decision (n_decisions + 1 if not protected)
                        as GeoTIFFs, NaN outside the planning units
    species_protection.csv
                        per species: metric class, mean protected range and population
                        fractions across replicates
    greedy_baseline.npz, summary.json
                        the cells with the highest protpop value (with populations
                        static the protpop reward is a sum over cells, so this plan
                        maximizes it) and the reward of policy vs. greedy plans

    python predict_sharks.py --metric IUCN --setup protpop [--replicates 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shark_common as sc
import torch
from rasterio.transform import Affine


def write_tif(path: Path, grid2d: np.ndarray, grid_info: dict) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid2d.shape[0],
        width=grid2d.shape[1],
        count=1,
        dtype="float32",
        crs=grid_info["crs"],
        transform=Affine(*grid_info["transform"]),
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(grid2d.astype(np.float32), 1)


def term_rewards(episode) -> np.ndarray:
    """Raw value of each reward term in the current state of the environment."""
    return np.array([obj.calc_reward(episode.env) for obj in episode.rewards._reward_obj_list])


def species_fractions(env) -> tuple[np.ndarray, np.ndarray]:
    """Protected fraction of each species' range (cells) and population."""
    range_cells = (env.h > 0).float().sum(dim=1).clamp(min=1)
    prot_range = (env.protected_range / range_cells).cpu().numpy()
    prot_pop = (env.protected_population / env.ext_risk.init_pop.clamp(min=1e-12)).cpu().numpy()
    return prot_range, prot_pop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sc.add_config_arguments(parser)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or sc.default_run_name(args)
    run_dir = sc.MODELS_DIR / run_name
    with open(run_dir / "settings.json") as f:
        settings = json.load(f)
    if settings["config"] != sc.config_settings(args):
        raise ValueError(
            f"Arguments do not match the configuration {run_name} was trained with:\n"
            f"  trained: {settings['config']}\n  now:     {sc.config_settings(args)}"
        )
    weight_files = sorted((run_dir / "weights_history").glob("epoch_*.npy"))[-args.replicates :]
    if not weight_files:
        raise FileNotFoundError(f"No weights in {run_dir / 'weights_history'}")
    if len(weight_files) < args.replicates:
        print(f"Warning: only {len(weight_files)} weight snapshots available")
    n_rep = len(weight_files)

    out_dir = run_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    data = sc.load_prepared(args.max_species, args.subset_seed)
    episode = sc.create_episode_runner(args, data, save_protection_history=True)
    env = episode.env
    print(sc.describe_runner(episode))
    grid_shape = data["grid_shape"]
    rows, cols = data["rows"], data["cols"]
    status = env.ext_risk.init_status.cpu().numpy()
    reward_names = episode.rewards.names

    freq = np.zeros(grid_shape)
    decision_sum = np.zeros(len(rows))
    prot_range_all, prot_pop_all, rewards_all, plans = [], [], [], []
    for i, wf in enumerate(weight_files):
        info, _ = episode.run_episode(np.load(wf))
        protection = np.nan_to_num(info["protection_matrix"][0]).astype(np.float32)
        # protection_history counts, per cell, the decisions after which it was protected
        hist = info["protection_history"].cpu().numpy()
        n_done = hist.max()
        decision_pu = np.where(hist > 0, n_done - hist + 1, 0)
        decision = np.zeros(grid_shape, dtype=np.int16)
        decision[rows, cols] = decision_pu
        prot_range, prot_pop = species_fractions(env)
        rewards = info["rewards"].cpu().numpy()
        np.savez_compressed(
            out_dir / f"replicate_{i:03d}.npz",
            protection_matrix=protection,
            decision=decision,
            protected_range_fraction=prot_range,
            protected_pop_fraction=prot_pop,
            species=data["names"],
            reward_names=np.array(reward_names),
            rewards=rewards,
            weights_file=wf.name,
        )
        freq += protection
        decision_sum += np.where(decision_pu > 0, decision_pu, args.n_decisions + 1)
        prot_range_all.append(prot_range)
        prot_pop_all.append(prot_pop)
        rewards_all.append(rewards)
        plans.append(protection[rows, cols] > 0)
        print(
            f"  replicate {i + 1}/{n_rep} ({wf.name}): {int(protection.sum())} cells, "
            f"rewards {dict(zip(reward_names, np.round(rewards, 4).tolist(), strict=True))}"
        )

    freq /= n_rep
    tag = f"CAPTAIN3_{args.metric}_{args.setup}"
    if args.max_species is not None:
        tag += f"_sub{args.max_species}"
    pd.DataFrame({"PUID": np.arange(1, freq.size + 1), "Priority": freq.ravel()}).to_csv(
        out_dir / f"{tag}_averaged_budget{sc.TARGET_FRACTION}_replicates{n_rep}.csv",
        index=False,
    )
    freq_tif = np.full(grid_shape, np.nan)
    freq_tif[rows, cols] = freq[rows, cols]
    write_tif(out_dir / "protection_frequency.tif", freq_tif, data["grid"])
    mean_dec = np.full(grid_shape, np.nan)
    mean_dec[rows, cols] = decision_sum / n_rep
    write_tif(out_dir / "mean_decision.tif", mean_dec, data["grid"])

    prot_range_all, prot_pop_all = np.array(prot_range_all), np.array(prot_pop_all)
    pd.DataFrame(
        {
            "species": data["names"],
            f"{args.metric}_class": status + 1,
            "protected_range_fraction_mean": prot_range_all.mean(0),
            "protected_range_fraction_sd": prot_range_all.std(0),
            "protected_pop_fraction_mean": prot_pop_all.mean(0),
        }
    ).to_csv(out_dir / "species_protection.csv", index=False)

    # Greedy reference: the cells with the highest protpop value (optimal for protpop)
    target = episode.budget_manager.total_target
    value = sc.protpop_cell_value(data["values"], status)
    greedy_idx = np.argsort(-value, kind="stable")[:target]
    episode.rewards.reset()
    env.reset()
    env.update_protection_matrix(torch.as_tensor(greedy_idx, device=env.device))
    greedy_rewards = term_rewards(episode)
    greedy = np.zeros(grid_shape, dtype=np.float32)
    greedy[rows[greedy_idx], cols[greedy_idx]] = 1
    g_range, g_pop = species_fractions(env)
    np.savez_compressed(
        out_dir / "greedy_baseline.npz",
        protection_matrix=greedy,
        protected_range_fraction=g_range,
        protected_pop_fraction=g_pop,
        species=data["names"],
        reward_names=np.array(reward_names),
        rewards=greedy_rewards,
    )

    greedy_mask = np.zeros(len(rows), dtype=bool)
    greedy_mask[greedy_idx] = True
    overlap = [float((p & greedy_mask).sum() / target) for p in plans]
    rewards_all = np.array(rewards_all)
    summary = {
        "run_name": run_name,
        "replicates": n_rep,
        "target_cells": int(target),
        "reward_names": reward_names,
        "policy_rewards_mean": rewards_all.mean(0).tolist(),
        "policy_rewards_sd": rewards_all.std(0).tolist(),
        "greedy_protpop_rewards": greedy_rewards.tolist(),
        "overlap_with_greedy_mean": float(np.mean(overlap)),
        "species_below_10pct_range_protected_mean": float((prot_range_all < 0.1).sum(1).mean()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to {out_dir}")


if __name__ == "__main__":
    main()
