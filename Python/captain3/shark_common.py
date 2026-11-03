"""Shared setup of the CAPTAIN 3 (captain-ng) shark prioritization runs.

Re-implements the CAPTAIN 2 analyses (Python/captain_script_and_data/predict_configs):
static SDMs, no costs or disturbance, 10% of the planning units protected, and species
prioritized by one of three metrics (IUCN, FUSE, EDGE2), each coded 1-5 in the species
table (5 = highest priority).

Episodes have a single time step without population dynamics: the protection target
is reached through N_DECISIONS decisions, with the features recomputed after each one.

Two reward setups:
    protpop        CalcRewardProtectedPopulation: per metric class, the mean protected
                   fraction of the species' populations, weighted 1/8/16/32/64 for
                   classes 1-5. Same objective as the CAPTAIN 2 "species_risk" reward
                   (pareto_mlt mode, risk_weights -64 -32 -16 -8 -1), up to a constant.
    protpop_worst  protpop + CalcRewardWorstSpecies: mean protected fraction of the
                   worst-served 10% of species (soft maximin, as in the AF corridor
                   runs). Needs a reward calibration (calibrate_sharks.py).

Data and outputs: the SDMs are read from R/Data/tif files continental and the species
table from R/Data; prepare_shark_data.py writes the preprocessed SDMs to
<SHARK_RUNS_DIR>/prepared. SHARK_RUNS_DIR defaults to Python/captain3/runs (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

import captain as cn  # noqa: E402

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parents[2]
SDM_DIR = REPO_DIR / "R" / "Data" / "tif files continental"
TRAIT_FILE = (
    REPO_DIR
    / "R"
    / "Data"
    / "continental_shark_conservation_metrics_10_harmonised_IUCN_categories.csv"
)
SPECIES_COL = "Species name"
RUNS_ROOT = Path(
    os.environ.get("SHARK_RUNS_DIR", REPO_DIR / "Python" / "captain3" / "runs")
).expanduser()
PREPARED_DIR = RUNS_ROOT / "prepared"
CALIBRATION_DIR = RUNS_ROOT / "calibration"
MODELS_DIR = RUNS_ROOT / "models"  # one folder per trained run

METRICS = ("IUCN", "FUSE", "EDGE2")
SETUPS = ("protpop", "protpop_worst")

# -----------------------------------------------------------------------------
# Settings carried over from CAPTAIN 2 (predict_configs/shark_predict_*.txt)
# -----------------------------------------------------------------------------
FULL_SUITABILITY = 0.9  # full_suitability: suitability above this is set to 1
MIN_SUITABILITY = 0.5  # prob_threshold: below this a cell is not in the species range
TARGET_FRACTION = 0.1  # protection_target: fraction of planning units protected
N_DECISIONS = 20  # protection_time_steps: decisions to reach the target
# Weight of metric classes 1..5 (CAPTAIN 2 risk_weights -1 ... -64, LC ... CR)
CLASS_WEIGHTS = np.array([1.0, 8.0, 16.0, 32.0, 64.0])
N_CLASSES = 5

# -----------------------------------------------------------------------------
# CAPTAIN 3 settings
# -----------------------------------------------------------------------------
# Individuals per cell at suitability 1. Populations are static and all rewards are
# protected fractions, so this only sets the scale of the population features.
SPECIES_K = 100.0
WORST_FRACTION = 0.1  # share of species averaged by the worst-served reward
HIDDEN_DIM = [32, 16]
CONVOLUTION = 5  # neighbourhood of protection_matrix_conv (radius 4 cells)
LEARNING_RATE = 0.2  # initial ES learning rate (x0.99 per epoch), as in the AF runs
INITIAL_SIGMA = 0.3
DEFAULT_SEED = 20251103
CONFIG_VERSION = 1  # bump when the environment / features change


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def captain_commit() -> str:
    """Commit (and dirty flag) of the captain-ng checkout in use."""
    root = Path(cn.__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--", "captain"], text=True
        ).strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_prepared(max_species: int | None = None, subset_seed: int = 1) -> dict:
    """Load the preprocessed SDMs written by prepare_shark_data.py.

    Args:
        max_species: Optional random subset of species (for quick local tests). The
            planning units are then the cells where any of these species occurs.
        subset_seed: Seed of the species subset.

    Returns:
        Dict with values (n_species, n_cells) float32, names, rows, cols (grid
        coordinates of the planning units, row-major order), grid_shape and grid
        (transform / CRS of the rasters).
    """
    values_file = PREPARED_DIR / "sdm_values.npy"
    if not values_file.exists():
        raise FileNotFoundError(f"{values_file} not found: run prepare_shark_data.py first")
    meta = np.load(PREPARED_DIR / "sdm_meta.npz")
    with open(PREPARED_DIR / "grid.json") as f:
        grid = json.load(f)
    values = np.load(values_file, mmap_mode="r")
    names = meta["names"].astype(str)
    rows, cols = meta["rows"], meta["cols"]

    if max_species is not None and max_species < len(names):
        rng = np.random.default_rng(subset_seed)
        idx = np.sort(rng.choice(len(names), max_species, replace=False))
        values = np.asarray(values[idx])
        names = names[idx]
        keep = values.max(axis=0) > 0
        values, rows, cols = values[:, keep], rows[keep], cols[keep]
    else:
        values = np.asarray(values)

    return {
        "values": np.ascontiguousarray(values, dtype=np.float32),
        "names": names,
        "rows": rows,
        "cols": cols,
        "grid_shape": tuple(int(i) for i in meta["grid_shape"]),
        "grid": grid,
    }


def load_species_status(metric: str, names: np.ndarray) -> np.ndarray:
    """Metric class of each species as 0..4 (0 = class 1, lowest priority)."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    with warnings.catch_warnings():
        # species without SDM (or outside a --max-species subset) are dropped, as intended
        warnings.filterwarnings("ignore", message=".*entries in DataFrame were not found")
        traits = cn.data_loader.load_trait_table(TRAIT_FILE, names, ref_column=SPECIES_COL)
    values = traits[metric].to_numpy()
    if np.any(pd.isna(values)) or not np.isin(values, np.arange(1, N_CLASSES + 1)).all():
        raise ValueError(f"{metric} values must be integers 1-{N_CLASSES}")
    return values.astype(np.int64) - 1


def class_weights_per_species_count(status: np.ndarray) -> np.ndarray:
    """Per-class weights of CalcRewardProtectedPopulation giving the CAPTAIN 2 reward.

    CAPTAIN 2 averaged the protected fraction within each class and weighted the class
    means; CalcRewardProtectedPopulation sums species, so each class weight is divided
    by its number of species (absent classes get weight 0).
    """
    counts = np.bincount(status, minlength=N_CLASSES).astype(float)
    return np.where(counts > 0, CLASS_WEIGHTS / np.maximum(counts, 1), 0.0)


def log_scale(x: np.ndarray) -> np.ndarray:
    """log10 of positive values rescaled to [0.05, 1]; zeros stay 0."""
    out = np.zeros_like(x, dtype=np.float32)
    pos = x > 0
    if pos.any():
        lx = np.log10(x[pos])
        span = lx.max() - lx.min()
        out[pos] = 0.05 + 0.95 * ((lx - lx.min()) / span if span > 0 else 1.0)
    return out


def static_features(values: np.ndarray, status: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Static per-cell features describing what the protected population reward pays for.

    rarity_class<k>: sum over the species of class k of suitability / range total, i.e.
        the share of each species' population held by the cell (with protected-fraction
        rewards a cell is worth ~1/range for each species in it).
    protpop_value: the protpop reward gained by protecting the cell (the class-weighted
        sum of the above), the same for every decision since populations are static.
    All log-scaled to [0, 1].
    """
    totals = values.sum(axis=1)
    onehot = np.zeros((N_CLASSES, len(status)), dtype=np.float32)
    onehot[status, np.arange(len(status))] = 1.0
    rarity = (onehot / totals[None, :]) @ values  # (n_classes, n_cells)
    weights = class_weights_per_species_count(status)
    protpop_value = weights @ rarity
    feats = np.vstack([*(log_scale(r) for r in rarity), log_scale(protpop_value)])
    names = [f"rarity_class{k + 1}" for k in range(N_CLASSES)] + ["protpop_value"]
    return feats.astype(np.float32), names


def protpop_cell_value(values: np.ndarray, status: np.ndarray) -> np.ndarray:
    """protpop reward of protecting each cell alone (rewards add up over cells)."""
    totals = values.sum(axis=1)
    weights = class_weights_per_species_count(status)
    per_species = weights[status] / totals
    return (per_species @ values) / (weights[status]).sum()


# -----------------------------------------------------------------------------
# Configuration, run folders and calibration files
# -----------------------------------------------------------------------------
def config_settings(args) -> dict:
    """Settings that define the environment, features and reward terms.

    Two runs with the same config can share a reward calibration.
    """
    return {
        "config_version": CONFIG_VERSION,
        "metric": args.metric,
        "setup": args.setup,
        "max_species": args.max_species,
        "subset_seed": args.subset_seed if args.max_species is not None else None,
        "n_decisions": args.n_decisions,
        "target_fraction": TARGET_FRACTION,
        "full_suitability": FULL_SUITABILITY,
        "min_suitability": MIN_SUITABILITY,
        "class_weights": CLASS_WEIGHTS.tolist(),
        "worst_fraction": WORST_FRACTION if args.setup == "protpop_worst" else None,
        "hidden_dim": HIDDEN_DIM,
    }


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


def calibration_file(args) -> Path:
    config = config_settings(args)
    return (
        CALIBRATION_DIR
        / f"reward_calibration_{args.metric}_{args.setup}_{config_hash(config)}.json"
    )


def default_run_name(args) -> str:
    name = f"{args.metric}_{args.setup}"
    if args.max_species is not None:
        name += f"_sub{args.max_species}"
    return name


def add_config_arguments(parser) -> None:
    """Command line arguments shared by calibration, training and prediction."""
    parser.add_argument("--metric", required=True, choices=METRICS)
    parser.add_argument("--setup", required=True, choices=SETUPS)
    parser.add_argument(
        "--n-decisions",
        type=int,
        default=N_DECISIONS,
        help="decisions to reach the target, features recomputed in between",
    )
    parser.add_argument(
        "--max-species", type=int, default=None, help="random subset of species, for quick tests"
    )
    parser.add_argument("--subset-seed", type=int, default=1)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


# -----------------------------------------------------------------------------
# Environment and episode runner
# -----------------------------------------------------------------------------
def create_episode_runner(
    args,
    data: dict | None = None,
    save_protection_history: bool = False,
) -> cn.EpisodeRunner:
    """Build environment, features, policy, rewards and budget for one runner."""
    device = args.device or default_device()
    if data is None:
        data = load_prepared(args.max_species, args.subset_seed)
    values, names = data["values"], data["names"]
    rows, cols, grid_shape = data["rows"], data["cols"], data["grid_shape"]
    n_species, n_cells = values.shape

    status = load_species_status(args.metric, names)

    # Planning units: cells where at least one species reaches MIN_SUITABILITY
    mask = np.full(grid_shape, np.nan, dtype=np.float32)
    mask[rows, cols] = 1.0
    grid = np.zeros((n_species, *grid_shape), dtype=np.float32)
    grid[:, rows, cols] = values
    sdms = cn.SpatialData(
        data=grid,
        mask=mask.copy(),
        lower_bound=0,
        upper_bound=1,
        min_threshold=MIN_SUITABILITY,
        names=list(names),
    )
    del grid
    zeros = np.zeros(grid_shape, dtype=np.float32)
    disturbance = cn.SpatialData(data=zeros, mask=mask.copy(), lower_bound=0, upper_bound=1)
    costs = cn.SpatialData(data=zeros, mask=mask.copy(), lower_bound=0, upper_bound=1)
    protection = cn.SpatialData(data=zeros, mask=mask.copy(), lower_bound=0, upper_bound=1)

    env = cn.BioEnv(
        sdms=sdms,
        disturbance=disturbance,
        costs=costs,
        protection_matrix=protection,
        species_k=np.full(n_species, SPECIES_K),
        growth_rates=np.ones(n_species),
        sensitivity_rates=np.zeros((n_species, 1)),
        # populations are static (env_dynamics=False): no dispersal matrix needed
        cached_dispersal_matrix=sp.identity(n_cells, dtype=np.float32, format="csr"),
        ext_risk=cn.ExtinctionRiskStatic(init_status=status, n_classes=N_CLASSES),
        device=device,
    )

    static, static_names = static_features(values, status)
    dynamic = ["species_richness", "total_population", "current_ext_risk", "protection_matrix_conv"]
    if args.setup == "protpop_worst":
        dynamic.append("worst_species_value")
    feature_extractor = cn.FeatureExtractor(
        env,
        feature_set=dynamic,
        static_features=static,
        static_features_names=static_names,
        convolution=CONVOLUTION,
        device=device,
        worst_fraction=WORST_FRACTION,
    )

    model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=HIDDEN_DIM)
    policy = cn.PolicyNetwork(model, seed=args.seed, device=device)

    reward_terms = [
        cn.CalcRewardProtectedPopulation(
            name="protpop",
            threat_weights=class_weights_per_species_count(status),
            range_exponent=1.0,
            device=device,
        )
    ]
    reward_weights = [1.0]
    if args.setup == "protpop_worst":
        reward_terms.append(cn.CalcRewardWorstSpecies(name="worst", worst_fraction=WORST_FRACTION))
        reward_weights.append(getattr(args, "worst_weight", 1.0))
    rewards = cn.Rewards(
        reward_obj_list=reward_terms,
        reward_weights=np.array(reward_weights),
        cumulative_reward=False,  # episode reward = final state
    )

    target = int(np.floor(TARGET_FRACTION * n_cells))
    budget_manager = cn.GlobalBudgetManager(
        total_target=target,
        cells_per_time_step=target,
        feature_updates_per_time_step=args.n_decisions,
    )

    return cn.EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        budget_manager=budget_manager,
        n_steps=1,
        env_dynamics=False,
        save_protection_history=save_protection_history,
    )


def describe_runner(episode: cn.EpisodeRunner) -> str:
    env = episode.env
    counts = env.ext_risk.species_per_class().cpu().numpy().astype(int)
    return (
        f"{env.n_species} species, {env.n_cells} planning units, target "
        f"{episode.budget_manager.total_target} cells in "
        f"{episode.budget_manager.feature_updates_per_time_step} decisions of "
        f"<= {episode.budget_manager.cells_per_step}; species per class 1-5: {counts.tolist()}; "
        f"{episode.feature_extractor.n_features} features "
        f"({', '.join(episode.feature_extractor.feature_names)}); "
        f"rewards {episode.rewards.names}"
    )
