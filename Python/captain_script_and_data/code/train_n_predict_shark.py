# load CAPTAIN and dependencies
import copy
import os, csv
import glob
import sys
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=3)
import captain as cn
# import seaborn as sns
# import matplotlib.pyplot as plt
from captain.utilities import empirical_data_parser as cn_util
from captain.utilities import sdm_utils as sdm_utils
from captain.biodivsim import SimGrid as cn_sim_grid
from captain.utilities import misc as cn_misc
from captain.utilities.misc import parse_str
# import rioxarray as rxr
import sparse
import argparse
import configparser

cmd_line = True

if cmd_line:
    p = argparse.ArgumentParser()
    p.add_argument('config_file', type=str, help='config file')
    args = p.parse_args()
    config_file = args.config_file
else:
    print("A config file must be provided.")
    sys.exit("\n")
config = configparser.ConfigParser()
config.read(config_file)
SEED = parse_str(config["general"]["seed"])

PRIORITY_VAR = config["policy"]["priority_variable"]

#-------- SETTINGS ---------#

# policy settings (objectives and reward)
ext_risk_class = cn.ExtinctioRiskProtectedRangeFuture # cn.ExtinctioRiskProtectedRange
budget = parse_str(config["policy"]["budget"])  # if budget = None: np.sum(costs) * protection_fraction
reward = "pareto_mlt"

r_w = parse_str(config["policy"]["reward_weights"])
r_w = np.round(r_w / np.sum(r_w), 2)
rewards_variables = parse_str(config["policy"]["rewards_variables"])
reward_weights = my_dict = {key: value for key, value in zip(rewards_variables, r_w)}


# paths, output files
data_wd = config["files"]["data_dir"]
models_wd = config["files"]["models_dir"]
env_data_wd = os.path.join(data_wd, config["files"]["env_data_dir"])

results_wd = os.path.join(data_wd, config["files"]["result_dir"] + "_" + str(SEED))
try: os.mkdir(results_wd)
except FileExistsError: pass

max_species_cutoff = parse_str(config["general"]["max_species_cutoff"])

parse_str(config["policy"]["feature_set"])


#-------- ANALYSIS ---------#



# env_init GENERATOR
def generate_init_env(config, seed, batch_n):
    # -------- LOAD DATA ---------#

    # load present and future SDMs
    sdms, species_names = sdm_utils.get_data_from_list(
        os.path.join(data_wd, config["files"]["sdm_data_dir"]),
        tag="/*.tif", max_species_cutoff=None)

    # load species traits (migration, sensitivity, growth, conservation status)
    trait_tbl = pd.read_csv(os.path.join(data_wd, config["files"]["trait_tbl_file"]))

    if max_species_cutoff is not None:
        rnd_gen = cn_util.get_rnd_gen(seed=seed + batch_n)
        rnd_ind = rnd_gen.choice(range(trait_tbl.shape[0]), max_species_cutoff, replace=False)
        sdms = sdms[rnd_ind]
        species_names = np.array(species_names)[rnd_ind]
        trait_tbl = trait_tbl.iloc[rnd_ind]
        # trait_tbl = trait_tbl.head(max_species_cutoff)
        trait_tbl, species_col_name = cn_misc.match_taxa(trait_tbl, species_names)
        print(species_names[:5])

    if np.all(np.array(species_names) == trait_tbl.iloc[:, 0].to_numpy()):
        pass
    else:
        print("\nWarning! Some taxa names do not match!\n")
        species_names = np.array(np.array(species_names))
        print(species_names[(species_names == trait_tbl.iloc[:, 0].to_numpy()) == 0])
        sys.exit("exiting.")

    n_species = sdms.shape[0]

    empirical_sensitivity = np.zeros(n_species)

    conservation_status = 5 - np.array(trait_tbl[PRIORITY_VAR])
    species_carbon = np.ones(n_species)

    # load disturbance and cost
    disturbance_layer = np.zeros(sdms[0].shape)
    cost_layer = np.zeros(sdms[0].shape)
    selective_disturbance_layer = None

    # -------- ENVIRONMENT SETUP ---------#

    # create env
    sdms[sdms > parse_str(config["env_settings"]["full_suitability"])] = 1

    # make it a graph without gaps
    suitability = cn_util.get_habitat_suitability(sdms, parse_str(config["env_settings"]["prob_threshold"]))
    original_grid_shape, reference_grid_pu, reference_grid_pu_nan, xy_coords, graph_coords = sdm_utils.get_graph(
        suitability)
    max_species_richness = np.nanmax(np.nansum(suitability, axis=0))

    # set up parameters
    if parse_str(config["species_settings"]["use_empirical_dispersal_rates"]):
        emp_dispersal_rates = np.array(trait_tbl['dispersal_ability'])
        dispersal_rates = emp_dispersal_rates / np.nanmean(
            emp_dispersal_rates) * parse_str(config["species_settings"]["mean_dispersal_rate"])

    elif parse_str(config["species_settings"]["variable_dispersal_rates"]):
        dispersal_rates = np.exp(np.linspace(-2, 2, None))  # [::-1]
        dispersal_rates = dispersal_rates / np.mean(
            dispersal_rates) * parse_str(config["species_settings"]["mean_dispersal_rate"])

    else:
        dispersal_rates = parse_str(config["species_settings"]["mean_dispersal_rate"])

    # dispersal threshold (in number of cells)
    fname_sparse = "disp_probs_sp%s_th%s_batch%s.npz" % (n_species,
                                                         config["species_settings"]["dispersal_threshold"], batch_n)

    dispersal_probs = cn_sim_grid.dispersalDistancesThresholdCoord(
        length=graph_coords.shape[1], lambda_0=1, lat=graph_coords[1], lon=graph_coords[0],
        threshold=parse_str(config["species_settings"]["dispersal_threshold"]))
    dispersal_probs_sparse = sparse.COO(dispersal_probs)
    sparse.save_npz(os.path.join(data_wd, fname_sparse), dispersal_probs_sparse)

    # graph transform
    # graph SDMs
    graph_sdms, n_pus, grid_length = sdm_utils.grid_to_graph(sdms, reference_grid_pu)
    graph_suitability = cn_util.get_habitat_suitability(
        graph_sdms, parse_str(config["env_settings"]["prob_threshold"]),
        integer=parse_str(config["env_settings"]["round_habitat_suitability"]))
    graph_sdms_f = graph_sdms + 0
    graph_future_suitability = cn_util.get_habitat_suitability(
        graph_sdms_f, parse_str(config["env_settings"]["prob_threshold"]),
        integer=parse_str(config["env_settings"]["round_habitat_suitability"]))
    # how much the present suitability has to change per step to reach "future_suitability" in STEPS
    # can be multiplied by e.g. 0.5 to go only halfway to "future_suitability" in STEPS
    if parse_str(config["general"]["use_future_sdms"]):
        t = parse_str(config["general"]["time_to_future_suitability"])
        delta = (graph_future_suitability - graph_suitability) * 1 / t
        delta_suitability_per_step = {'delta': delta,
                                      'threshold': parse_str(config["env_settings"]["prob_threshold"])}
    else:
        delta_suitability_per_step = {'delta': 0, 'threshold': parse_str(config["env_settings"]["prob_threshold"])}
        graph_sdms_f = graph_sdms + 0
        graph_future_suitability = graph_suitability + 0

    if parse_str(config["policy"]["add_to_existing_protected_areas"]):
        mpas_layer_name = parse_str(config["files"]["protection_file"])
        if parse_str(config["files"]["protection_file"]) is not None:
            existing_mpas, _ = sdm_utils.load_map(
                os.path.join(env_data_wd, parse_str(config["files"]["protection_file"])))
            disturbance_in_protect = (existing_mpas > 0) * disturbance_layer
            # reset disturbance in MPAs to 0
            disturbance_layer[existing_mpas > 0] = 1 - parse_str(config["policy"]["max_protection_level"])
            graph_protection_matrix, _, __ = sdm_utils.grid_to_graph(existing_mpas > 0, reference_grid_pu, n_pus,
                                                                     nan_to_zero=True)
            existing_protected = sdm_utils.graph_to_grid(graph_protection_matrix, reference_grid_pu, zero_to_nan=True)
            existing_protection_fraction = np.sum(graph_protection_matrix[graph_protection_matrix > 0]) / n_pus
        else:
            sys.exit("protection_file not specified")
    else:
        graph_protection_matrix = None
        existing_protection_fraction = 0
        existing_mpas = None

    graph_disturbance, _, __ = sdm_utils.grid_to_graph(disturbance_layer, reference_grid_pu, n_pus, nan_to_zero=True)
    if selective_disturbance_layer is not None:
        graph_selective_disturbance, _, __ = sdm_utils.grid_to_graph(selective_disturbance_layer, reference_grid_pu,
                                                                     n_pus,
                                                                     nan_to_zero=True)
    else:
        graph_selective_disturbance = graph_disturbance

    if parse_str(config["general"]["use_cost"]):
        graph_cost, _, __ = sdm_utils.grid_to_graph(cost_layer, reference_grid_pu, n_pus, nan_to_zero=True)
    else:
        graph_cost, _, __ = sdm_utils.grid_to_graph(np.ones(reference_grid_pu.shape) * 0.01, reference_grid_pu, n_pus,
                                                    nan_to_zero=True)

    protection_target = parse_str(config["general"]["protection_target"])
    if parse_str(config["policy"]["budget"]) is None:
        # budget set as a function of costs and target protection fraction
        if graph_protection_matrix is None:
            budget = np.sum(graph_cost) * protection_target
        else:
            budget = np.sum(graph_cost) * (protection_target - existing_protection_fraction)

    #-----
    # STEP UPDATES
    target_protected_cells = int(np.floor(protection_target * n_pus))
    time_to_protect = parse_str(config["general"]["protection_time_steps"])
    cells_to_be_protected_per_time_step = target_protected_cells / time_to_protect

    if parse_str(config["general"]["feature_update_frequency"]) is None:
        cells_to_be_protected_per_observe_step = cells_to_be_protected_per_time_step
    else:
        cells_to_be_protected_per_observe_step = np.minimum(
            parse_str(config["general"]["feature_update_frequency"]), cells_to_be_protected_per_time_step)

    total_observe_steps = int(parse_str(config["general"]["steps"]) * (
            target_protected_cells / cells_to_be_protected_per_observe_step))

    observe_steps_per_time_step = int(cells_to_be_protected_per_time_step / cells_to_be_protected_per_observe_step)
    cells_to_be_protected_per_observe_step = int(cells_to_be_protected_per_observe_step)


    #--------------
    size_protection_unit = np.array([1, 1])  # only 1x1 makes sense in graph-grid!
    r_seeds_dict = {
        'rnd_sensitivity': 1 * seed,
        'rnd_growth': 2 * seed,
        'rnd_config_policy': 3 * seed,
        'rnd_k_species': 4 * seed,
        'rnd_dispersal': 5 * seed
    }


    rs_g = cn.get_rnd_gen(r_seeds_dict['rnd_growth'])
    if parse_str(config['species_settings']['variable_growth_rates']):
        growth_rates = rs_g.beta(0.5, 0.5, n_species) + 1
        growth_rates = growth_rates / np.mean(growth_rates) * 2.
    else:
        growth_rates = 2.

    if parse_str(config["env_settings"]["use_K_species"]):
        # species-specific carrying capacities
        rs_k = cn.get_rnd_gen(r_seeds_dict['rnd_k_species'])
        if parse_str(config["env_settings"]["species_k_rnd_uniform"]):
            K_species = 2 + rs_k.random(n_species) * 100
        else:
            K_species = 2 + rs_k.beta(0.6, 1.2, n_species) * 100
        h3d = K_species[:, np.newaxis, np.newaxis] * graph_suitability
    else:
        sys.exit("hd3 object not initialized (option use_K_species = False not available)")

    # create SimGrid object: initGrid with empirical 3d histogram
    stateInitializer = cn.EmpiricalStateInit(h3d)

    max_K_cell = np.max(h3d) * 2 # max K per cell set > than highest based on K species (only K species matters)
    K_max = (h3d.sum((0)) > 0) * max_K_cell

    # initialize grid to reach carrying capacity: no disturbance, no sensitivity
    env2d = cn.BioDivEnv(budget=0.1,
                         gridInitializer=stateInitializer,
                         length=grid_length,
                         n_species=n_species,
                         K_max=K_max,
                         disturbance_sensitivity=np.zeros(n_species),
                         disturbanceGenerator=cn.FixedEmpiricalDisturbanceGenerator(0),
                         # to fast forward to w 0 disturbance
                         dispersal_rate=5,
                         growth_rate=[2],
                         resolution=size_protection_unit,
                         habitat_suitability=graph_suitability,
                         cost_pu=graph_cost,
                         precomputed_dispersal_probs=dispersal_probs_sparse,
                         use_small_grid=True,
                         K_species=K_species
                         )

    # evolve system to reach K_max
    env2d.set_calc_reward(False)
    env2d.fast_forward(parse_str(config["env_settings"]["steps_fast_fw"]),
                       disturbance_effect_multiplier=0, verbose=False, skip_dispersal=True)

    if parse_str(config["general"]["do_plots"]):
        species_richness_init = sdm_utils.graph_to_grid(env2d.bioDivGrid.speciesPerCell(), reference_grid_pu)
        cn_util.plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                         cmap="YlGnBu", show=False, title="Species richness (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "species_richness_natural_FWD.png"), dpi=250)

        pop_density = sdm_utils.graph_to_grid(env2d.bioDivGrid.individualsPerCell(), reference_grid_pu)
        cn_util.plot_map(pop_density, z=reference_grid_pu_nan, nan_to_zero=False,
                         cmap="Greens", show=False, title="Population density (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "population_density_natural_FWD.png"), dpi=250)

    if parse_str(config["files"]["plot_env_states"]):
        cn.plot_env_state(env2d, wd=results_wd, species_list=[])

    # make empirically rare / threatened species sensitive
    sensitivities = {
        'disturbance_sensitivity': empirical_sensitivity,
        'selective_sensitivity': empirical_sensitivity,
        'climate_sensitivity': np.zeros(n_species)
    }

    # set extinction risks
    ext_risk_obj = ext_risk_class(
        natural_state=env2d.grid_obj_previous,
        current_state=env2d.bioDivGrid,
        starting_rl_status=conservation_status,
        evolve_status=parse_str(config["extinction_risk"]["evolve_rl_status"]),
        # relative_pop_thresholds=relative_pop_thresholds,
        epsilon=0.5,
        # eps=1: last change, eps=0.5: rolling average, eps<0.5: longer legacy of long-term change
        sufficient_protection=0.5,
        pop_decrease_threshold=parse_str(config["extinction_risk"]["pop_decrease_threshold"]),
        min_individuals_cell=parse_str(config["extinction_risk"]["min_individuals_cell"]),
        relative_protected_range_thresholds=parse_str(config["extinction_risk"]["relative_protected_range_thresholds"]),
        risk_weights=parse_str(config["extinction_risk"]["risk_weights"]),
        min_protected_cells=parse_str(config["extinction_risk"]["min_protected_cells"]))


    d = np.einsum('sxy->xy', h3d)
    protection_steps = np.round(
        (d[d > 1].size * parse_str(config["general"]["protection_target"])) / (
                size_protection_unit[0] * size_protection_unit[1])).astype(int)
    print("Protection steps:", protection_steps)

    mask_disturbance = (K_max > 0).astype(int)
    disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
    selective_disturbanceGenerator = cn.FixedEmpiricalDisturbanceGenerator(0)
    init_disturbance = disturbanceGenerator.updateDisturbance(
        graph_disturbance * mask_disturbance * (1 - parse_str(config["env_settings"]["zero_disturbance"])))
    init_selective_disturbance = selective_disturbanceGenerator.updateDisturbance(
        graph_selective_disturbance * mask_disturbance * (1 - parse_str(config["env_settings"]["zero_selective_disturbance"])))

    # config simulation
    config_obj = cn.ConfigOptimPolicy(rnd_seed=r_seeds_dict['rnd_config_policy'],
                                  obsMode=1,
                                  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
                                  feature_update_per_step=True,
                                  steps=total_observe_steps,
                                  simulations=1,
                                  observePolicy=1,  # 0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
                                  disturbance=-1,
                                  degrade_steps=5,
                                  initial_disturbance=init_disturbance,  # set initial disturbance matrix
                                  initial_selective_disturbance=init_selective_disturbance,
                                  initial_protection_matrix=graph_protection_matrix,
                                  edge_effect=0,
                                  protection_cost=1,
                                  n_nodes=[2, 2],
                                  random_sim=0,
                                  # "0: fixed (replicable) simulations; 1: random; 2: fixed training, seq pickle"
                                  rewardMode=reward,
                                  obs_error=0,  # "Amount of error in species counts (feature extraction)"
                                  use_true_natural_state=True,
                                  resolution=size_protection_unit,
                                  grid_size=env2d.length,
                                  budget=budget,
                                  dispersal_rate=dispersal_rates,
                                  growth_rates=growth_rates,  # can be 1 values (list of 1 item) or or one value per species
                                  use_climate=0,  # "0: no climate change, 1: climate change, 2: climate disturbance,
                                  # 3: climate change + random variation"
                                  rnd_alpha=0,
                                  # (st.dev of sp.-specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
                                  outfile=config["files"]["outfile"] + config["general"]["seed"] + ".log",
                                  # model settings
                                  trained_model=None,
                                  temperature=1,
                                  deterministic_policy=1,  # 0: random policy (altered by temperature);
                                  # 1: deterministic policy (overrides temperature)
                                  sp_threshold_feature_extraction=0.001,
                                  start_protecting=1,
                                  plot_sim=False,
                                  plot_species=[],
                                  wd_output=results_wd,
                                  grid_h=env2d.bioDivGrid.h,  # 3D hist of species (e.g. empirical)
                                  distb_objects=[disturbanceGenerator, selective_disturbanceGenerator],
                                  # list of distb_obj, selectivedistb_obj
                                  return_env=True,
                                  ext_risk_obj=ext_risk_obj,
                                  # here set to 1 because env2d.bioDivGrid.h is already fast-evolved to carrying capcaity
                                  max_K_multiplier=1,
                                  suitability=graph_suitability,
                                  future_suitability=graph_future_suitability,
                                  delta_suitability_per_step=delta_suitability_per_step,
                                  cost_layer=graph_cost,
                                  actions_per_step=observe_steps_per_time_step,
                                  heuristic_policy="random",
                                  minimize_policy=False,
                                  use_empirical_setup=True,
                                  max_protection_level=parse_str(config["policy"]["max_protection_level"]),
                                  dynamic_print=True,
                                  precomputed_dispersal_probs=dispersal_probs_sparse,
                                  use_small_grid=True,
                                  sensitivities=sensitivities,
                                  K_species=K_species,
                                  pre_steps=10,
                                  sp_carbon=species_carbon,
                                  reference_grid_pu=reference_grid_pu
                                  )

    config_init = copy.deepcopy(config_obj)
    config_init.steps = 5
    config_init.budget = 0
    config_init.actions_per_step = 1
    env_init = cn.run_restore_policy(config_init)
    env_init.set_budget(config_obj.budget)
    config_obj.observe_steps_per_time_step = observe_steps_per_time_step
    config_obj.cells_to_be_protected_per_observe_step = cells_to_be_protected_per_observe_step
    config_obj.reference_grid_pu_nan = reference_grid_pu_nan
    config_obj.target_protected_cells = target_protected_cells
    return env_init, config_obj


i = 0
# train model
if config["general"]["run_mode"] == "train":
    envList = []
    for i in range(int(config["general"]["batch_size"])):
        print("Setup batch n. %s" % i)
        env_train, config_train = generate_init_env(config, SEED * i, batch_n=i)
        env_train._verbose = i == 0 # only job 0 is verbose
        env_train.rewardMode = reward
        env_train.feature_set = parse_str(config["policy"]["feature_set"])
        env_train.iterations =  parse_str(config["general"]["steps"])  * config_train.observe_steps_per_time_step + 1
        # if budget is None:
        env_train.budget = config_train.budget
        env_train.set_max_n_protected_cells(config_train.target_protected_cells)
        env_train.reset_init_values()
        env_train.set_calc_reward(True)
        env_train._reward_min_protection = None
        envList.append(copy.deepcopy(env_train))

    if parse_str(config["general"]["fine_tune"]):
        trained_model_file = os.path.join(models_wd, config["files"]["model_file_name"])
        wNN_params = cn_misc.get_nn_params_from_file(trained_model_file,
                                                     load_best_epoch=False,
                                                     sample_from_iteration=None,
                                                     seed=SEED)
    else:
        wNN_params = None

    tmp_res_plot_class = cn_util.plot_map_class(z=config_train.reference_grid_pu_nan, fig_s=[10, 5.5])

    env_tmp = cn.runBatchGeneticStrategyEmpirical(envList,
                                                  epochs=parse_str(config["general"]["epochs"]),
                                                  lr=0.5,
                                                  lr_adapt=0.01,
                                                  temperature=1,
                                                  max_workers=0,
                                                  outfile=config["files"]["outfile"] + config["general"]["seed"] + ".log",
                                                  obsMode=1,
                                                  observe_error=0,
                                                  running_reward_start=-1000,
                                                  eps_running_reward=0.5,
                                                  sigma=0.2,
                                                  wNN=wNN_params,
                                                  n_NN_nodes=[3, 2],
                                                  increase_temp=0,
                                                  deterministic=True,
                                                  resolution=np.array([1, 1]),
                                                  max_temperature=1000,
                                                  sp_threshold_feature_extraction=1,
                                                  wd_output=results_wd,
                                                  actions_per_step=config_train.observe_steps_per_time_step,
                                                  protection_per_step=config_train.cells_to_be_protected_per_observe_step,
                                                  plot_res_class=tmp_res_plot_class,
                                                  reward_weights=reward_weights,
                                                  return_env=True,
                                                  skip_dispersal=True
                                                  )

# predict model
elif config["general"]["run_mode"] == "predict":
    summary_file = os.path.join(results_wd, config["files"]["outfile"] + "_summary_stats.txt")
    with open(summary_file, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        head = ["Seed", "Disturbance", "RestoredCells",
                "CostRestoredCells", "Budget_left",
                "Carbon", "TotPopSize",
                "ExtantSpecies", "ExtinctSpecies",
                "MSA", "PDF", "STAR-t", "STAR-r",
                "CR", "EN", "VU", "NT", "LC",
                "MSA_init", "PDF_init", "STAR-t_init", "STAR-r_init",
                "CR_init", "EN_init", "VU_init", "NT_init", "LC_init",
                "Mean_pr", "Min_pr", "Max_pr",
                "non_protected_range_sp", "non_protected_pop_sp",
                "CR_pr_init_rl", "EN_pr_init_rl", "VU_pr_init_rl", "NT_pr_init_rl", "LC_pr_init_rl",
                "CR_pr", "EN_pr", "VU_pr", "NT_pr", "LC_pr",
                "Reward"]
        writer.writerow(head)

    protect_m = []


    for rep_i in range(parse_str(config["general"]["prediction_runs"])):

        # get NN weights
        print("\nSetup batch n. %s of %s" % (rep_i + 1, parse_str(config["general"]["prediction_runs"])))
        wNN_params = cn_misc.get_nn_params_from_file(
            os.path.join(models_wd, config["files"]["model_file_name"]),
            load_best_epoch=False,
            sample_from_iteration=parse_str(config["files"]["sample_weights_from_last_iter"]),
            seed=SEED + rep_i)

        envList = []
        env_train, config_train = generate_init_env(config, SEED * rep_i, batch_n=rep_i)
        env_train._verbose = True # only job 0 is verbose
        env_train.rewardMode = reward
        env_train.feature_set = parse_str(config["policy"]["feature_set"])
        env_train.iterations =  parse_str(config["general"]["steps"])  * config_train.observe_steps_per_time_step + 1
        # if budget is None:
        env_train.budget = config_train.budget
        env_train.set_max_n_protected_cells(config_train.target_protected_cells)
        env_train.reset_init_values()
        env_train.set_calc_reward(True)
        env_train._reward_min_protection = None
        envList.append(copy.deepcopy(env_train))


        tmp_res_plot_class = cn_util.plot_map_class(z=config_train.reference_grid_pu_nan, fig_s=[10, 5.5])

        env_list_optim = cn.runBatchGeneticStrategyEmpirical(envList,
                                                      epochs=1,
                                                      lr=0.5,
                                                      lr_adapt=0.01,
                                                      temperature=1,
                                                      max_workers=0,
                                                      outfile=config["files"]["outfile"] + config["general"]["seed"] + ".log",
                                                      obsMode=1,
                                                      observe_error=0,
                                                      running_reward_start=-1000,
                                                      eps_running_reward=0.5,
                                                      sigma=0.2,
                                                      wNN=wNN_params,
                                                      n_NN_nodes=[3, 2],
                                                      increase_temp=0,
                                                      deterministic=True,
                                                      resolution=np.array([1, 1]),
                                                      max_temperature=1000,
                                                      sp_threshold_feature_extraction=1,
                                                      wd_output=results_wd,
                                                      actions_per_step=config_train.observe_steps_per_time_step,
                                                      protection_per_step=config_train.cells_to_be_protected_per_observe_step,
                                                      plot_res_class=tmp_res_plot_class,
                                                      reward_weights=reward_weights,
                                                      return_env=True,
                                                      skip_dispersal=True
                                                      )
        #
        
        
        
        

        env_optim = env_list_optim[0].env
        env_init = envList[0]
        print("Init risk_label_counts:", env_init.risk_label_counts())
        print("Optim risk_label_counts:", env_optim.risk_label_counts())

        species_richness_init = sdm_utils.graph_to_grid(env_optim.bioDivGrid.speciesPerCell(), env_optim.reference_grid_pu)
        pop_density = sdm_utils.graph_to_grid(env_optim.bioDivGrid.individualsPerCell(), env_optim.reference_grid_pu)
        protection_matrix = sdm_utils.graph_to_grid(env_optim.bioDivGrid.protection_matrix, env_optim.reference_grid_pu)


        ## OTHER STATS
        # calc final metrics
        m = cn.calc_MSA(env_optim.grid_obj_previous.h, env_optim.bioDivGrid.h)
        p = cn.calc_PDF_from_grid(env_optim.grid_obj_previous.individualsPerSpecies(),
                                  env_optim.bioDivGrid.individualsPerSpecies())

        st, sr = cn.calc_STAR_from_grid(env_optim.grid_obj_previous.h, env_optim.bioDivGrid.h,
                                        sp_natural_range=env_optim.grid_obj_previous.geoRangePerSpecies(),
                                        sp_ext_risk=env_optim.getExtinction_risk_labels(),
                                        quandrant_grid_indx=env_optim._quandrant_grid_indx)

        t = np.mean(st[st > 0])
        r = np.mean(sr[sr > 0])

        print("""
             MSA: %s
             PDF: %s
             STAR-t: %s
             STAR-r: %s

             """ % (m, p, t, r)
              )

        print("Init risk_label_counts:", env_init.risk_label_counts())
        print("Final risk_label_counts:", env_optim.risk_label_counts())

        # calc init stats
        m_init = cn.calc_MSA(env_init.grid_obj_previous.h, env_init.bioDivGrid.h)
        p_init = cn.calc_PDF_from_grid(env_init.grid_obj_previous.individualsPerSpecies(),
                                       env_init.bioDivGrid.individualsPerSpecies())

        st, sr = cn.calc_STAR_from_grid(env_init.grid_obj_previous.h, env_init.bioDivGrid.h,
                                        sp_natural_range=env_init.grid_obj_previous.geoRangePerSpecies(),
                                        sp_ext_risk=env_init.getExtinction_risk_labels(),
                                        quandrant_grid_indx=env_init._quandrant_grid_indx)

        t_init = np.mean(st[st > 0])
        r_init = np.mean(sr[sr > 0])
        #
        protected_range_fraction = env_optim.bioDivGrid.protectedRangePerSpecies() / env_optim.bioDivGrid.geoRangePerSpecies()
        # average fraction of protected range per threat class (initial classification)
        protected_range_risk_class_init = np.zeros(5)
        for i in range(5):
            if i in env_optim.species_risk_criteria.starting_rl_status:
                protected_range_risk_class_init[i] = np.mean(
                    protected_range_fraction[env_optim.species_risk_criteria.starting_rl_status == i])

        protected_range_risk_class_current = np.zeros(5)
        rl = env_optim.getExtinction_risk_labels()
        for i in range(5):
            if i in rl:
                protected_range_risk_class_current[i] = np.mean(protected_range_fraction[rl == i])


        # get number of non protected species

        pm = env_optim.bioDivGrid.protection_matrix
        h = env_optim.bioDivGrid.h
        h_nat = env_optim.grid_obj_previous.h
        h_non_protected = (h > 1) * (1 - pm)
        n_nonprotected_species = np.einsum('sxy->xy', h_non_protected)
        h_protected = (h > 1) * pm
        protected_area_per_species = env_optim.bioDivGrid.protectedRangePerSpecies()
        min_number_protected_cells_sp = 1
        non_protected_range_species_id = protected_area_per_species < min_number_protected_cells_sp

        # n species with no protection
        n_nonprotected_range_species_total = np.sum(non_protected_range_species_id)
        protected_pop_per_species = env_optim.bioDivGrid.protectedIndPerSpecies()
        n_nonprotected_pop_species = len(protected_pop_per_species[protected_pop_per_species < 1])

        protected_ranges = [np.mean(protected_range_fraction),
                            np.min(protected_range_fraction),
                            np.max(protected_range_fraction),
                            n_nonprotected_range_species_total,
                            n_nonprotected_pop_species] + list(protected_range_risk_class_init) + list(protected_range_risk_class_current)

        #
        rl = env_optim.getExtinction_risk_labels()

        protected_pop_per_species = env_optim.bioDivGrid.protectedIndPerSpecies()
        n_non_protected_species = len(protected_pop_per_species[protected_pop_per_species < 1])
        # REWARD
        # weighted by class
        pop_size = env_optim.bioDivGrid.individualsPerSpecies()
        rl[pop_size == 0] = 0
        # fraction of non protected range
        pr_fr = np.zeros(env_optim.n_species)
        pr_fr[pop_size > 0] = 1 - (protected_pop_per_species[pop_size > 0] / pop_size[pop_size > 0])
        pr_fr = 1 - (protected_pop_per_species / pop_size)

        avg_protection_rl_class = np.ones(5)
        for i in range(5):
            if i in rl:
                avg_protection_rl_class[i] = 1 - np.mean(pr_fr[rl == i])
        reward_species = np.sum(avg_protection_rl_class * -env_optim.species_risk_criteria.risk_weights)
        reward_species = reward_species / np.sum(-env_optim.species_risk_criteria.risk_weights) * 100
        #----
        reward_disturbance = 0
        tot_reward = reward_species + reward_disturbance
        #

        rl = env_optim.getExtinction_risk_labels()
        h_non_protected = (env_optim.bioDivGrid.h > 1) * (1 - env_optim.bioDivGrid.protection_matrix)
        h_protected = (env_optim.bioDivGrid.h > 1) * env_optim.bioDivGrid.protection_matrix
        protected_area_per_species = np.einsum('sxy -> s', h_protected)  # shape: species
        min_number_protected_cells_sp = 1
        non_protected_species_id = protected_area_per_species < min_number_protected_cells_sp
        # ---

        with open(summary_file, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            stats_i = np.array(
                [rep_i,
                 np.mean(env_optim.bioDivGrid.disturbance_matrix[env_optim.bioDivGrid.disturbance_matrix > 0]),
                 env_optim.bioDivGrid.protection_matrix[env_optim.bioDivGrid.protection_matrix > 0].size,
                 env_optim.cost_protected_quadrants, env_optim.budget,
                 np.log10(env_optim.current_carbon), np.log10(np.sum(env_optim.bioDivGrid.h)),
                 env_optim.bioDivGrid.numberOfSpecies(),
                 env_init.bioDivGrid.numberOfSpecies() - env_optim.bioDivGrid.numberOfSpecies(),
                 m, p, t, r] + list(env_optim.risk_label_counts()) + [
                    m_init, p_init, t_init, r_init] + list(env_init.risk_label_counts()) + protected_ranges + [
                    tot_reward])

            writer.writerow(list(np.nan_to_num(stats_i)))

        current_sp_range = env_optim.bioDivGrid.geoRangePerSpecies()
        current_pop_sizes = env_optim.bioDivGrid.individualsPerSpecies(
            parse_str(config["extinction_risk"]["min_individuals_cell"]))
        current_protected_pop_sizes = env_optim.bioDivGrid.protectedIndPerSpecies(
            parse_str(config["extinction_risk"]["min_individuals_cell"]))
        current_protected_range = env_optim.bioDivGrid.protectedRangePerSpecies()

        x = np.array([current_protected_range, env_optim.species_risk_criteria.population_change]).T

        population_change_to_init = current_pop_sizes / env_optim.species_risk_criteria._init_pop_size - 1

        np.savez_compressed(
            file=os.path.join(results_wd, config["files"]["outfile"] + config["general"][
                "seed"] + "_stats_%s.npz" % rep_i),
            protection_matrix=protection_matrix,
            protected_range_fraction=protected_range_fraction,
            simulated_init_status=env_init.getExtinction_risk_labels(),
            resulting_status=env_optim.getExtinction_risk_labels(),
            history=np.array(env_optim.history),
            history_var_names=env_optim.history_var_names,
            current_sp_range=current_sp_range,
            current_pop_sizes=current_pop_sizes,
            current_protected_pop_sizes=current_protected_pop_sizes,
            current_protected_range=current_protected_range

        )

        protect_m.append(protection_matrix)

    nzp = np.array(protect_m)
    np.save(os.path.join(results_wd, config["files"]["outfile"] + config["general"][
                "seed"] + "_protection_matrix.npy"), nzp)
    print("Results saved in:", results_wd)

    priority = np.einsum('rxy -> xy', nzp)


    cn_util.plot_map(priority, z=config_train.reference_grid_pu_nan, nan_to_zero=False,
                     cmap="GnBu", show=False, title="",
                     outfile=os.path.join(results_wd, config["files"]["outfile"] + config["general"][
                "seed"] + "_protection_priorities.png"), dpi=350)

else:
    raise ValueError("Invalid run mode")


