import sys, os, glob
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from .misc import get_rnd_gen, print_update
from .empirical_data_parser import get_habitat_suitability, generate_h3d, plot_map
from . import sdm_utils
import sparse, copy
from ..biodivsim.StateInitializer import EmpiricalStateInit
from ..biodivsim.BioDivEnv import BioDivEnv
from ..biodivsim.DisturbanceGenerator import FixedEmpiricalDisturbanceGenerator
from ..algorithms.runOptimizedRestorePolicy import run_restore_policy, ConfigOptimPolicy
from ..biodivsim import SimGrid as cn_sim_grid

def rnd_init_h(grid_shape,
               n_groups,
               seed=None,
               gaussian=False,
               single_cell=False,
               min_threshold=0.0,
               fg_init_k=None,
               sig=0.1,
               round_to_int=True):

    rs = get_rnd_gen(seed)
    if single_cell:
        h = np.zeros((n_groups, grid_shape[0], grid_shape[1]))
        rx = rs.choice(grid_shape[0], n_groups, replace=True)
        ry = rs.choice(grid_shape[1], n_groups, replace=True)
        for i in range(n_groups):
            h[i, rx[i], ry[i]] = 1
    elif gaussian:
        h_list = []
        for i in range(n_groups):
            h = np.zeros(grid_shape)
            length = h.shape[0]
            n_peaks = 3
            scale = length * sig
            indx = np.meshgrid(np.arange(length), np.arange(length))
            locsxy = rs.uniform(0, length, (2, n_peaks))
            for i in range(n_peaks):
                # print(locsxy[:,i])
                h += scipy.stats.norm.pdf(
                    indx[0], loc=locsxy[0, i], scale=scale
                ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, i], scale=scale)
            h_list.append(h)

        h = np.array(h_list)
    else:
        h = rs.random((n_groups, grid_shape[0], grid_shape[1]))

    h[h < min_threshold] = 0
    if fg_init_k is not None:
        h = np.einsum("sxy, s -> sxy", h, 1 / np.max(h, axis=(1, 2)))
        h = np.einsum("sxy, s -> sxy", h, fg_init_k)
        if round_to_int:
            h = np.round(h).astype(int)
    return h


def generate_conditional_sdms(seed, n_species, original_shape,
                              probs, reference_grid_pu,
                              env_layers_z, env_layers_f_z, coord_grid_z,
                              dispersal = [0.05, 1]):
    size = np.prod(original_shape)

    sdms = np.zeros((n_species, original_shape[0], original_shape[1]))
    sdms_f = np.zeros(sdms.shape)

    # make up tolerance distributions (MVN) including lon/lat
    rs_sp_init = get_rnd_gen(seed)
    i = 0
    while i != n_species:
        # lon, lat = rs_sp_init.integers(ORIGINAL_SHAPE)
        indx = rs_sp_init.choice(range(size), p=probs.flatten(), size=1)
        x, y = np.unravel_index(indx, shape=original_shape)
        lon, lat = x[0], y[0]

        print_update("sim species %s / %s" % (i + 1, n_species))
        if reference_grid_pu[lon, lat]:
            # if on the included area (eg land)
            m = env_layers_z[:, lon, lat]
            s = rs_sp_init.uniform(0.05, 1, len(m))
            dens = scipy.stats.norm.pdf(env_layers_z,
                                        loc=m[:, np.newaxis, np.newaxis], scale=s[:, np.newaxis, np.newaxis])

            dens_f = scipy.stats.norm.pdf(env_layers_f_z,
                                          loc=m[:, np.newaxis, np.newaxis], scale=s[:, np.newaxis, np.newaxis])

            m = coord_grid_z[:, lon, lat]
            d = rs_sp_init.uniform(dispersal[0], dispersal[1])  # radius, dispersal rate
            dens_loc = (coord_grid_z - m[:, np.newaxis, np.newaxis]) ** 2
            distance = np.sqrt(np.sum(dens_loc, axis=0))
            dens_loc = - np.exp(-d * distance)
            dens_loc = dens_loc / np.nanmax(dens_loc)

            dens = dens / np.nanmax(dens, axis=(1, 2))[:, np.newaxis, np.newaxis]
            dens_f = dens_f / np.nanmax(dens_f, axis=(1, 2))[:, np.newaxis, np.newaxis] # ! normalize by present SDM

            sdm_i = dens_loc * np.prod(dens, axis=0)
            sdms[i] = sdm_i / np.nanmax(sdm_i)

            sdm_i_f = dens_loc * np.prod(dens_f, axis=0)
            sdms_f[i] = sdm_i_f / np.nanmax(sdm_i_f)
            i += 1

    return sdms, sdms_f


def generate_conditional_disturbance(seed, original_shape,
                                     n_peaks=3,
                                     scale_range=[0.05, 0.2],
                                     max_disturbance=0.99):
    rs = get_rnd_gen(seed)
    h = np.zeros(original_shape)
    length = h.shape[0]
    scale = length * rs.uniform(scale_range[0], scale_range[1], n_peaks)
    indx = np.meshgrid(np.arange(length), np.arange(length))
    locsxy = rs.uniform(0, length, (2, n_peaks))
    for i in range(n_peaks):
        d = scipy.stats.norm.pdf(
            indx[0], loc=locsxy[0, i], scale=scale[i]
        ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, i], scale=scale[i])
        h += d / np.nanmax(d)

    h[h > max_disturbance] = max_disturbance
    return h





def plot_sdm(sdms, sdms_f, i):
    fig = plt.figure(figsize=(10, 5))
    fig.tight_layout()
    fig.add_subplot(1, 2, 1)
    sns.heatmap(sdms[i], cmap='viridis')
    plt.gca().set_title("Species %s (present)" % i)
    fig.add_subplot(1, 2, 2)
    sns.heatmap(sdms_f[i], cmap='viridis')
    plt.gca().set_title("Species %s (future)" % i)
    plt.show()





def plot_rl_by_class(files, ylims=None, outname="", rl_lab=None):

    if rl_lab is None:
        rl_lab = ['carbon',
                  # 'carbon_in_protected_area', 'net_carbon',
                  # 'cost',
                  # 'population',
                  # 'protected_species',  # 'MSA',
                  'CR', 'EN', 'VU', 'NT', 'LC']


    status_list = []
    for i in range(len(files)):
        f = files[i]

        res = dict(np.load(f))
        # [k for k in res.keys()]

        n_species = len(res['protected_range_fraction'])
        history = np.array(res['history'])
        history_var_names = res['history_var_names']

        status_indx = [j for j in range(len(history_var_names)) if history_var_names[j] in ['CR', 'EN', 'VU', 'NT', 'LC']]
        history[:, np.array(status_indx)] *= n_species

        status_indx = [list(history_var_names).index(j) for j in rl_lab]
        status = history[:, np.array(status_indx)]

        status_list.append(status)

    status_list = np.array(status_list)

    p = [
        # carbon,
        "#993404",
        #  cost, population, MSA
        # "#bcbddc", "#969696","#fd8d3c" ,"#66c2a4",
        # RL status
        "#DA4741", "#EDAA4C", "#F4EB5A", "#BBD25B", "#468351"]
    fig = plt.figure(figsize=(20*0.6, 10*0.6))
    if outname is not None:
        outfile = os.path.join(os.path.dirname(f), "results_%s.pdf" % outname)
    else:
        outfile = None

    for rl in range(len(rl_lab)):
        rl_name = rl_lab[rl]
        fig.add_subplot(2,3,rl + 1)
        time_bins = np.arange(history.shape[0]) + 2020

        status = status_list[:, :, rl]

        if rl_name == 'carbon':
            status = (status - 1) * 100

        pred = np.median(status, axis=0)

        if len(files) > 1:
            edge_width = 0.25
            plt.fill_between(time_bins,
                             y1=np.max(status, axis=0).T,
                             y2=np.min(status, axis=0).T,
                             step="pre",
                             color=p[rl],
                             alpha=0.2)

            plt.fill_between(time_bins,
                             y1=np.quantile(status, q=0.975, axis=0).T,
                             y2=np.quantile(status, q=0.025, axis=0).T,
                             step="pre",
                             color=p[rl],
                             alpha=0.2)

            plt.fill_between(time_bins,
                             y1=np.quantile(status, q=0.75, axis=0).T,
                             y2=np.quantile(status, q=0.25, axis=0).T,
                             step="pre",
                             color=p[rl],
                             alpha=0.2)
        else:
            edge_width = 1

        # if rl == 2:
        #     plt.gca().set_title(
        #         outname, fontweight="bold", fontsize=12
        #     )
        plt.plot(time_bins, pred.T, "o-", color=p[rl], mec="k", mew=edge_width)
        if ylims is not None:
            plt.ylim(bottom=ylims[rl][0], top=ylims[rl][1])
        plt.xlabel("Year")
        if rl_name in ['CR', 'EN', 'VU', 'NT', 'LC']:
            plt.ylabel("Number of %s species" % rl_name)
        elif rl_name == 'carbon':
            plt.ylabel("Net carbon (%)")
        else:
            plt.ylabel("Total %s" % rl_name)
        # plt.show()

    if outfile is not None:
        fig.tight_layout()
        fig.savefig(outfile, dpi=250)
        plt.close()
        print("saved", outfile)






# env_init GENERATOR
def generate_init_env(rnd_seed,
                      n_species,
                      original_shape,
                      reference_grid_pu,
                      reference_grid_pu_nan,
                      graph_coords,
                      env_names_lab,
                      env_layers,
                      env_layers_z,
                      env_layers_f_z,
                      coord_grid_z,
                      round_habitat_suitability,
                      dispersal_threshold,
                      use_future_sdms,
                      time_to_future_suitability,
                      data_wd,
                      results_wd,
                      budget_reference,
                      protection_target,
                      n_pus,
                      ext_risk_class,
                      reward,
                      sp_dispersal_rates,
                      elevation_w = 1,
                      mean_carb=None,
                      growth_rate_mul=1,
                      outfile_tag="",
                      actions_per_step = 1,
                      minimize_policy = False,
                      max_protection_level = 1,
                      degrade_steps = 5,
                      edge_effect = 0,
                      feature_update_per_step = True,
                      do_plots=False,
                      plot_sim = False,
                      dynamic_print = True,
                      force_recompute_dispersal = False,
                      ):
    # CREATE SDMs
    probs = (np.nan_to_num(env_layers[0]) * np.nan_to_num(env_layers[1])) ** 2
    # probs += np.nan_to_num(env_layers[3]) * elevation_w
    probs += (scipy.stats.beta.pdf(np.nan_to_num(env_layers[3]), 3, 3) * elevation_w)
    probs = probs / np.sum(probs)

    if do_plots:
        sns.heatmap(probs)
        plt.title('Prob. starting point')
        plt.show()

        sns.heatmap(env_layers[0])
        plt.title('Precipitation')
        plt.show()

    # generate present and future ranges
    sdms, sdms_f = generate_conditional_sdms(seed=rnd_seed,
                                             n_species=n_species,
                                             original_shape=original_shape,
                                             probs=probs,
                                             reference_grid_pu=reference_grid_pu,
                                             env_layers_z=env_layers_z,
                                             env_layers_f_z=env_layers_f_z,
                                             coord_grid_z=coord_grid_z)

    h = np.nansum(sdms, axis=(1, 2))
    present = np.sort(np.log(h))[::-1]
    h = np.nansum(sdms_f, axis=(1, 2))
    future = np.sort(np.log(h))[::-1]

    prob_threshold = 0.3  # truncate habitat suitability below
    full_suitability = 0.9  # above this -> set to 1
    sdms[sdms > full_suitability] = 1
    sdms_f[sdms_f > full_suitability] = 1

    suitability = get_habitat_suitability(sdms, prob_threshold)
    species_richness = np.sum(suitability, axis=0)
    max_species_richness = np.nanmax(species_richness)
    species_richness[species_richness == 0] = np.nan
    # species range size
    sp_range_size = np.nansum(sdms, axis=(1, 2))
    sp_range_size_rel = sp_range_size / np.max(sp_range_size)

    if do_plots:
        for i in range(300, 310):
            _ = plot_sdm(sdms, sdms_f, i)
        sns.lineplot(present)
        sns.lineplot(future)
        plt.title('Rank abundance (present & future)')
        plt.show()

        sns.heatmap(species_richness, cmap='coolwarm', vmin=0)
        plt.gca().set_title("Species richness")
        plt.show()

    # disturbance matrix (same functions as for SDMs)
    "set lower disturbance in high elevation"
    elevation = env_layers[env_names_lab == 'elevation'].squeeze()
    elevation[np.isnan(reference_grid_pu_nan)] = np.nan
    d = generate_conditional_disturbance(rnd_seed + 5 , original_shape, n_peaks=1000, scale_range=[0.01, 0.025])
    disturbance = d * np.exp(-elevation)

    if do_plots:
        sns.heatmap(disturbance, cmap='coolwarm')
        plt.gca().set_title("Disturbance")
        plt.show()

    # generate biomes
    "carbon should be higher in forest species - defined based on precipitation"
    "define cell type as in forest / grassland"
    biome = np.zeros(env_layers[0].shape)
    biome[np.isnan(env_layers[0])] = np.nan
    biome[env_layers[0] < 0.4] = 1
    biome[env_layers[0] < 0.25] = 2

    # ensure that all 'biomes' are sufficiently represented
    q2 = np.minimum(
        np.quantile(env_layers[0][np.isfinite(reference_grid_pu_nan)], .2),
        0.25)

    q1 = np.minimum(
        np.quantile(env_layers[0][np.isfinite(reference_grid_pu_nan)], .8),
        0.40)

    print("q1, q2,", q1, q2)
    biome = np.zeros(env_layers[0].shape)
    biome[np.isnan(env_layers[0])] = np.nan
    biome[env_layers[0] < q1] = 1
    biome[env_layers[0] < q2] = 2

    # calculate species range overlap with "biome"
    sp_range_size = np.nansum(sdms, axis=(1, 2))
    pr_biome = []
    for j in np.unique(biome):
        if np.isfinite(j):
            pr_biome.append(
                [np.nansum(sdms[i] * (biome == j)) / sp_range_size[i] for i in range(n_species)]
            )

    pr_biome_ = np.array(pr_biome).T
    pr_biome = np.einsum('sc, s -> sc', pr_biome_, 1 / np.sum(pr_biome_, 1))
    # habit = np.argmax(pr_biome, axis=1)  # 0: tree, 1: shrub, 2: herb
    rnd_gen_h = get_rnd_gen(rnd_seed + 9)
    # print("pr_biome[i, :].shape[0]", pr_biome[0, :].shape[0])
    # print(pr_biome)
    habit = np.array([rnd_gen_h.choice(range(pr_biome[i, :].shape[0]), p=pr_biome[i, :]) for i in range(pr_biome.shape[0])])

    if len(np.unique(habit)) < 3:
        rnd_ind = rnd_gen_h.choice(range(n_species),
                                   np.maximum(3, int(0.15 * n_species)),
                                   replace=False)
        habit[rnd_ind] = 1
        habit[:int(len(rnd_ind) / 3)] = 0
        habit[-int(len(rnd_ind) / 3):] = 2

    # growth rates dependent on habit
    rnd_gen = get_rnd_gen(rnd_seed + 2)
    scales = np.array([0.1, 2, 10]) * growth_rate_mul
    sp_growth_rate = 1 + (rnd_gen.weibull(0.75, n_species) * scales[habit])

    # carbon dependent of habit
    alphas = np.array([2., 2., 2.])
    if mean_carb is None:
        mean_carb = np.array([2., 1., 0.5])
    carbon_sp = rnd_gen.gamma(alphas[habit], mean_carb[habit] / alphas[habit])

    if do_plots:
        sns.heatmap(biome, cmap='YlGn_r')
        plt.gca().set_title("Biomes")
        plt.show()
        plt.scatter(habit, np.log(sp_growth_rate))
        plt.gca().set_title("Species habit vs growth rate")
        plt.show()


    # generate RL status, sensitivity, abundances
    sp_sensitivity = rnd_gen.beta(0.5, 0.5, n_species)
    sp_selective_sensitivity = np.zeros(n_species)
    sp_climate_sensitivity = rnd_gen.beta(0.5, 0.5, n_species)
    sensitivities = {
        'disturbance_sensitivity': sp_sensitivity,
        'selective_sensitivity': sp_selective_sensitivity,
        'climate_sensitivity': sp_climate_sensitivity
    }

    effect_disturbance = 1 - np.einsum('xy, s -> sxy',
                                       np.nan_to_num(disturbance), sp_sensitivity)

    sp_range_size_disturbed = np.einsum('sxy, sxy -> s',
                                        np.nan_to_num(sdms), effect_disturbance)

    risk_prob = 1 / (sp_range_size_rel ** 0.3 * (sp_range_size_disturbed / sp_range_size))
    risk_prob /= np.max(risk_prob)
    risk = 1 - risk_prob
    bins = ([0, 0.6, 0.8, 0.9, 0.95, 1])
    starting_rl_status = np.digitize(risk, bins) - 1
    # print("\nstarting_rl_status:", starting_rl_status, "\n")

    if do_plots:
        plt.scatter(sp_sensitivity, sp_range_size_disturbed / sp_range_size)
        plt.xlabel("sensitivity")
        plt.ylabel("fraction of range size")
        plt.show()

        plt.scatter(risk_prob, sp_range_size_disturbed / sp_range_size)
        plt.xlabel("risk_prob")
        plt.ylabel("fraction of range size")
        plt.show()

        plt.scatter(risk_prob, sp_range_size)
        plt.xlabel("risk_prob")
        plt.ylabel("species range size")
        plt.show()

        h = np.unique(starting_rl_status, return_counts=True)
        plt.bar(h[0], h[1],color=[ "#DA4741", "#EDAA4C", "#F4EB5A", "#BBD25B", "#468351"])
        plt.gca().set_title("Extinction risk")
        plt.show()

    pop_decrease_threshold = 0.01
    min_individuals_cell = 1

    relative_protected_range_thresholds = np.array([0.1,  # +1
                                                    0.25,  # +2
                                                    0.50,  # +3
                                                    0.80  # +4
                                                    ])

    risk_weights = np.array([-64, -32, -16, -8, -1])  # LC have a -1 so there is still reward in protecting them
    min_protected_cells = None

    # --- env settings
    max_K_cell = 10000
    max_K_multiplier = 10
    steps_fast_fw = 0

    # turn to graphs
    # graph SDMs
    graph_sdms, _, grid_length = sdm_utils.grid_to_graph(sdms, reference_grid_pu)
    graph_suitability = get_habitat_suitability(graph_sdms, prob_threshold,
                                                        integer=round_habitat_suitability)
    graph_sdms_f, _, __ = sdm_utils.grid_to_graph(sdms_f, reference_grid_pu)
    graph_future_suitability = get_habitat_suitability(graph_sdms_f, prob_threshold,
                                                               integer=round_habitat_suitability)
    # how much the present suitability has to change per step to reach "future_suitability" in STEPS
    # can be multiplied by e.g. 0.5 to go only halfway to "future_suitability" in STEPS


    if use_future_sdms:
        delta = (graph_future_suitability - graph_suitability) * 1 / time_to_future_suitability
        delta_suitability_per_step = {'delta': delta, 'threshold': prob_threshold}
    else:
        delta_suitability_per_step = {'delta': 0, 'threshold': prob_threshold}
        graph_sdms_f = graph_sdms + 0
        graph_future_suitability = graph_suitability + 0

    graph_disturbance, _, __ = sdm_utils.grid_to_graph(disturbance, reference_grid_pu)
    graph_selective_disturbance = graph_disturbance + 0
    # dispersal threshold (in number of cells)
    fname_sparse = "disp_probs_sp%s_c%s_th%s.npz" % (n_species, graph_disturbance.shape[0], dispersal_threshold)
    try:
        if force_recompute_dispersal:
            raise(ValueError)
        else:
            dispersal_probs_sparse = sparse.load_npz(os.path.join(data_wd, fname_sparse))
    except:
        dispersal_probs = cn_sim_grid.dispersalDistancesThresholdCoord(grid_length,
                                                                       lambda_0=1,
                                                                       lat=graph_coords[1],
                                                                       lon=graph_coords[0],
                                                                       threshold=dispersal_threshold)
        # np.save(os.path.join(data_wd, fname), dispersal_probs)
        dispersal_probs_sparse = sparse.COO(dispersal_probs)
        sparse.save_npz(os.path.join(data_wd, fname_sparse), dispersal_probs_sparse)
    # ---

    # set and rescale cost layer
    min_cost = 0.05
    c = generate_conditional_disturbance(rnd_seed + 6 , original_shape, n_peaks=100, scale_range=[0.02, 0.05])
    c = c ** 0.5
    f = disturbance * (c / np.mean(c))
    f_rescaled = f + np.abs(np.nanmin(f))
    f_rescaled[np.isnan(f_rescaled)] = np.abs(np.nanmin(f))
    f_rescaled = f_rescaled + min_cost
    cost_layer = f_rescaled / np.nanmax(f_rescaled)

    graph_cost, _, __ = sdm_utils.grid_to_graph(cost_layer, reference_grid_pu, n_pus, nan_to_zero=True)

    # budget set as a function of costs and target protection fraction
    if budget_reference is None:
        budget = np.sum(graph_cost) * protection_target * 2 # x2 to make sure the target is reached
    else:
        budget = budget_reference + 0


    # make somce cells unaffordable --> enforced in BioDivEnv.py: _enrichObs(self)
    "e.g. take out cells with the 5% highest disturbance"
    graph_cost[graph_disturbance > np.quantile(graph_disturbance.flatten(), 0.95)] = budget + 1

    if do_plots:
        sns.heatmap(sdm_utils.graph_to_grid(np.log(graph_cost + 1), reference_grid_pu, n_pus), cmap="Oranges")
        plt.gca().set_title("Costs")
        plt.show()

        c_tmp = graph_cost + 0
        c_tmp[graph_cost == np.max(graph_cost)] = np.nan
        sns.heatmap(sdm_utils.graph_to_grid(c_tmp, reference_grid_pu, n_pus), cmap="Purples")
        plt.gca().set_title("Costs (affordable)")
        plt.show()

    graph_protection_matrix = None

    K_biome = np.ones((grid_length, grid_length))  # * np.sum(graph_sdms, axis=0) <- K dependent of species richness
    # create n. individuals and carrying capacities
    mask_suitability = (np.sum(graph_sdms, axis=0) > 0).astype(int)
    # mask_suitability multiplies the disturbance that is therefore set to 0 where 0 species are found

    h3d, K_max, K_cells = generate_h3d(graph_sdms, graph_suitability, mask_suitability,
                                               max_K_cell, K_biome, max_K_multiplier)

    # make up K species (woody: fewer herbs: more)
    K_species_sorted = np.sort(2 + rnd_gen.beta(0.6, 1.2, n_species) * 100)
    K_species = np.zeros(n_species)
    K_species[habit == 0] = K_species_sorted[:len(habit[habit == 0])]
    K_species[habit == 1] = K_species_sorted[len(habit[habit == 0]):-len(habit[habit == 2])]
    K_species[habit == 2] = K_species_sorted[-len(habit[habit == 2]):]
    K_species_3D = K_species[:, np.newaxis, np.newaxis] * graph_suitability  # np.ones(h3d.shape)
    h3d = K_species_3D + 0

    if do_plots:
        plt.scatter(K_species, habit)
        plt.show()

    # create SimGrid object: initGrid with empirical 3d histogram
    stateInitializer = EmpiricalStateInit(h3d)
    size_protection_unit = np.array([1, 1])

    # initialize grid to reach carrying capacity: no disturbance, no sensitivity
    # print("\n\n\n carbon_sp", carbon_sp, np.sum(graph_disturbance), np.sum(h3d), "\n\n\n", )
    env2d = BioDivEnv(budget=0.1,
                      gridInitializer=stateInitializer,
                      length=grid_length,
                      n_species=n_species,
                      K_max=K_max,
                      disturbance_sensitivity=np.zeros(n_species),
                      disturbanceGenerator=FixedEmpiricalDisturbanceGenerator(0),
                      # to fast forward to w 0 disturbance
                      dispersal_rate=5,
                      growth_rate=[2],
                      resolution=size_protection_unit,
                      habitat_suitability=graph_suitability,
                      cost_pu=graph_cost,
                      precomputed_dispersal_probs=dispersal_probs_sparse,
                      use_small_grid=True,
                      K_species=K_species,
                      species_carbon_value=carbon_sp,
                      verbose=False
                      )

    # evolve system to reach K_max
    env2d.set_calc_reward(False)
    env2d.fast_forward(steps_fast_fw, disturbance_effect_multiplier=0, verbose=False, skip_dispersal=False)

    if do_plots:
        species_richness_init = sdm_utils.graph_to_grid(env2d.bioDivGrid.speciesPerCell(), reference_grid_pu)
        plot_map(species_richness_init, z=reference_grid_pu_nan, nan_to_zero=False, vmax=max_species_richness,
                         cmap="YlGnBu", show=False, title="Species richness (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "species_richness_natural_FWD.png"), dpi=250)

        pop_density = sdm_utils.graph_to_grid(env2d.bioDivGrid.individualsPerCell(), reference_grid_pu)
        plot_map(pop_density, z=reference_grid_pu_nan, nan_to_zero=False,
                         cmap="Greens", show=False, title="Population density (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "population_density_natural_FWD.png"), dpi=250)

        carbon_per_cell = sdm_utils.graph_to_grid(env2d.bioDivGrid.getCarbonValue_cell(), reference_grid_pu)
        plot_map(carbon_per_cell, z=reference_grid_pu_nan, nan_to_zero=False,
                         cmap="YlOrBr", show=False, title="Total carbon (Natural state - FWD)",
                         outfile=os.path.join(results_wd, "carbon_per_cell_FWD.png"), dpi=250)



    #
    # if do_plots:
    #     sns.heatmap(biome, cmap='YlGn_r')
    #     plt.gca().set_title("Biomes")
    #     plt.show()
    #     plt.scatter(habit, np.log(sp_growth_rate))
    #     plt.gca().set_title("Species habit vs growth rate")
    #     plt.show()
    #     sns.heatmap(np.einsum('sxy, s -> xy', sdms, carbon_sp), cmap='YlOrBr')
    #     plt.gca().set_title("Carbon")
    #     plt.show()
    #     sns.heatmap(np.einsum('sxy-> xy', sdms))
    #     plt.gca().set_title("Population density")
    #     plt.show()
    #


    # set extinction risks
    ext_risk_obj = ext_risk_class(natural_state=env2d.grid_obj_previous,
                                  current_state=env2d.bioDivGrid,
                                  starting_rl_status=starting_rl_status,
                                  evolve_status=True,
                                  # relative_pop_thresholds=relative_pop_thresholds,
                                  epsilon=0.5,
                                  # eps=1: last change, eps=0.5: rolling average, eps<0.5: longer legacy of long-term change
                                  sufficient_protection=0.5,
                                  pop_decrease_threshold=pop_decrease_threshold,
                                  min_individuals_cell=min_individuals_cell,
                                  relative_protected_range_thresholds=relative_protected_range_thresholds,
                                  risk_weights=risk_weights,
                                  min_protected_cells=min_protected_cells)


    d = np.einsum('sxy->xy', h3d)
    protection_steps = np.round(
        (d[d > 1].size * protection_target) / (size_protection_unit[0] * size_protection_unit[1])).astype(int)
    print("protection_steps:", protection_steps)


    mask_disturbance = (K_cells > 0).astype(int)
    disturbanceGenerator = FixedEmpiricalDisturbanceGenerator(0)
    selective_disturbanceGenerator = FixedEmpiricalDisturbanceGenerator(0)
    init_disturbance = disturbanceGenerator.updateDisturbance(graph_disturbance * mask_disturbance)
    init_selective_disturbance = selective_disturbanceGenerator.updateDisturbance(graph_selective_disturbance * mask_disturbance * 0)

    # config simulation
    config = ConfigOptimPolicy(rnd_seed=get_rnd_gen(rnd_seed),
                                  obsMode=1,
                                  # 0: random, 1: full monitor, 2: citizen-science, 3: one-time, 4: value, 5: area
                                  feature_update_per_step=feature_update_per_step,
                                  steps=protection_steps,
                                  simulations=1,
                                  observePolicy=1,  # 0: NO-OBSERVE-UPDATE 1: ORACLE 2: PROTECTATONCE
                                  disturbance=-1,
                                  degrade_steps=degrade_steps,
                                  initial_disturbance=init_disturbance,  # set initial disturbance matrix
                                  initial_protection_matrix=graph_protection_matrix,
                                  edge_effect=edge_effect,
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
                                  dispersal_rate=sp_dispersal_rates,
                                  growth_rates=sp_growth_rate,  # can be 1 values (list of 1 item) or or one value per species
                                  use_climate=0,  # "0: no climate change, 1: climate change, 2: climate disturbance,
                                  # 3: climate change + random variation"
                                  rnd_alpha=0,
                                  # (st.dev of sp.-specific fluctuation in mortality (if 'by_species' ==1 in SimpleGrid)
                                  outfile=outfile_tag + ".log",
                                  # model settings
                                  trained_model=None,
                                  temperature=1,
                                  deterministic_policy=1,  # 0: random policy (altered by temperature);
                                  # 1: deterministic policy (overrides temperature)
                                  sp_threshold_feature_extraction=0.001,
                                  start_protecting=1,
                                  plot_sim=plot_sim,
                                  plot_species=[],
                                  wd_output=results_wd,
                                  grid_h=env2d.bioDivGrid.h,  # 3D hist of species (e.g. empirical)
                                  distb_objects=[disturbanceGenerator, 0],
                                  # list of distb_obj, selectivedistb_obj
                                  return_env=True,
                                  ext_risk_obj=ext_risk_obj,
                                  # here set to 1 because env2d.bioDivGrid.h is already fast-evolved to carrying capcaity
                                  max_K_multiplier=1,
                                  suitability=graph_suitability,
                                  future_suitability=graph_future_suitability,
                                  delta_suitability_per_step=delta_suitability_per_step,
                                  cost_layer=graph_cost,
                                  actions_per_step=actions_per_step,
                                  heuristic_policy="random",
                                  minimize_policy=minimize_policy,
                                  use_empirical_setup=True,
                                  max_protection_level=max_protection_level,
                                  dynamic_print=dynamic_print,
                                  precomputed_dispersal_probs=dispersal_probs_sparse,
                                  use_small_grid=True,
                                  sensitivities=sensitivities,
                                  K_species=K_species,
                                  sp_carbon=carbon_sp,
                                  reference_grid_pu=reference_grid_pu,
                                  pre_steps=10

                                  )

    config_init = copy.deepcopy(config)
    config_init.steps = 5
    config_init.budget = 0
    config_init.actions_per_step = 1
    env_init = run_restore_policy(config_init)
    env_init.set_budget(config.budget)
    # env_init.species_carbon_value = carbon_sp
    # env_init.bioDivGrid.set_species_carbon_value(carbon_sp)
    # env_init.bioDivGrid.set_reference_grid_pu(reference_grid_pu)
    # env_init.grid_obj_most_recent._reference_grid_pu = reference_grid_pu
    return env_init, config_init

































