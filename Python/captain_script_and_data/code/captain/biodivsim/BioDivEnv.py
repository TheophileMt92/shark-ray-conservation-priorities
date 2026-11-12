import sys

import gym
from gym import spaces
import numpy as np
from scipy import ndimage
from enum import Enum
from ..biodivsim.SimGrid import SimGrid
from ..biodivsim.SpeciesRiskClass import ExtinctionRisk
import copy

np.set_printoptions(suppress=True, precision=3)
from ..agents.state_monitor import extract_features
import scipy
import random
from .DisturbanceGenerator import get_disturbance as get_disturbance
from ..agents.state_monitor import get_quadrant_indx_grid
from ..utilities.metrics import *
from ..utilities.misc import get_rnd_gen, print_update

DEBUG = 0
SMALL_NUMBER = 1e-13
class ActionType(Enum):
    Observe = 1
    Protect = 2
    Disturb = 3
    NoAction = 4
    NoObserve = 6  # only check which species are not protected based on old observation, no trends


class Action(object):

    PROTECT_COST = 0.2  # cost per cell
    OBSERVE_COST = 1  # currently not used
    NOOBSERVE_COST = 1  # currently not used

    def __init__(self, actionType: ActionType, value: int, value_quadrant: int):
        self.actionType = actionType
        self.value = value
        self.value_quadrant = value_quadrant


class ActionVec(object):

    PROTECT_COST = 0.2  # cost per cell
    OBSERVE_COST = 1  # currently not used
    NOOBSERVE_COST = 1  # currently not used

    def __init__(self, actionType: ActionType, value, value_quadrant):
        self.actionType = actionType
        self.value = value
        self.value_quadrant = value_quadrant


class smallGrid(object):
    def __init__(self, biodivgrid, include_future_h=False, env_layers=None):
        self._length = biodivgrid._length
        self._n_species = biodivgrid._n_species
        self._species_id = biodivgrid._species_id
        self._alpha = biodivgrid._alpha  # fraction killed (1 number)
        self._K_max = biodivgrid._K_max  # initial (max) carrying capacity
        self._lambda_0 = (
            biodivgrid._lambda_0
        )  # relative dispersal probability: always 1 at distance = 0
        self._growth_rate = (
            biodivgrid._growth_rate
        )  # potential number of offspring per individual per year at distance = 0
        self._disturbanceInitializer = biodivgrid._disturbanceInitializer
        self._disturbance_matrix = copy.deepcopy(biodivgrid._disturbance_matrix)
        self._K_cells = (1 - self._disturbance_matrix) * self._K_max
        self._K_disturbance_coeff = (
            biodivgrid._K_disturbance_coeff
        )  # if set to 0.5, K is 0.5*(1-disturbance)
        self._counter = biodivgrid._counter
        self._species_threshold = biodivgrid._species_threshold
        self._disturbance_sensitivity = (
            biodivgrid._disturbance_sensitivity
        )  # vector of sensitivity per species
        self._alpha_histogram = copy.deepcopy(biodivgrid._alpha_histogram)
        self._rnd_alpha = biodivgrid._rnd_alpha
        self._rnd_alpha_species = biodivgrid._rnd_alpha_species
        self._immediate_capacity = biodivgrid._immediate_capacity
        self._truncateToInt = biodivgrid._truncateToInt
        self._selective_disturbance_matrix = copy.deepcopy(
            biodivgrid._selective_disturbance_matrix
        )
        self._protection_matrix = copy.deepcopy(biodivgrid._protection_matrix)

        self._selectivedisturbanceInitializer = (
            biodivgrid._selectivedisturbanceInitializer
        )

        self._selective_sensitivity = copy.deepcopy(biodivgrid._selective_sensitivity)
        self._selective_alpha_histogram = copy.deepcopy(
            biodivgrid._selective_alpha_histogram
        )
        self._climate_sensitivity = copy.deepcopy(biodivgrid._climate_sensitivity)
        self._climate_as_disturbance = copy.deepcopy(biodivgrid._climate_as_disturbance)
        self._disturbance_dep_dispersal = copy.deepcopy(
            biodivgrid._disturbance_dep_dispersal
        )
        self._disturbance_matrix_diff = biodivgrid._disturbance_matrix_diff
        self._h = copy.deepcopy(biodivgrid._h)
        self._climate_layer = copy.deepcopy(biodivgrid._climate_layer)
        self._carbon_value_cell = biodivgrid.getCarbonValue_cell()
        self.species_carbon_value = biodivgrid.species_carbon_value
        self._K_species = biodivgrid._K_species
        self._K_species3D = biodivgrid._K_species3D
        self._species_threshold_per_cell = biodivgrid._species_threshold_per_cell
        self.future_h = biodivgrid.future_h
        self._reference_grid_pu = biodivgrid._reference_grid_pu
        self._rm_lingering_pops = biodivgrid._rm_lingering_pops
        self._n_pus = biodivgrid._n_pus
        self._env_layers = env_layers

    @property
    def length(self):
        return self._length

    @property
    def h(self):
        return self._h

    @property
    def protection_matrix(self):
        return self._protection_matrix

    def getSelectiveDisturbance(self):
        return self._selective_disturbance_matrix

    @property
    def disturbance_matrix(self):
        return self._disturbance_matrix

    @property
    def selective_disturbance_matrix(self):
        return self._selective_disturbance_matrix

    def individualsPerSpecies(self, min_individuals_cell=None):
        if min_individuals_cell is None:
            return np.einsum("sij->s", self._h)
        else:
            h_tmp = np.maximum(self._h - min_individuals_cell, 0)
            return np.einsum("sij->s", h_tmp)

    def protectedIndPerSpecies(self, min_individuals_cell=None):
        if min_individuals_cell is None:
            return np.einsum("sij, ij -> s", self._h, self._protection_matrix)
        else:
            h_tmp = np.maximum(self._h - min_individuals_cell, 0)
            return np.einsum("sij, ij -> s", h_tmp, self._protection_matrix)

    def individualsPerCell(self):
        return np.einsum("sij->ij", self._h)

    def speciesPerCell(self):
        presence_absence = self._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        return np.einsum("sij->ij", presence_absence)

    def geoRangePerSpecies(self):  # number of occupied cells
        # TODO clean up this: no need for temp, just return np.einsum('sij->s',self._h[ > 1]) ?
        temp = self._h + 0
        temp[temp > 1] = 1
        temp[temp < 1] = 0
        return np.einsum("sij->s", temp)

    def protectedRangePerSpecies(self):
        return np.einsum("sij, ij -> s", self._h >= self._species_threshold_per_cell,
                         self._protection_matrix)


    def histogram(self):
        return self._h

    def numberOfSpecies(self):
        return self._n_species

    def getCarbonValue_cell(self):
        return self._carbon_value_cell
    

# TODO: expose path to tree files (trees are otherwise simulated if not found)
# can be passed through the tree_generator argument in BioDivEnv()
def get_phylo_generator(seed=0, n_species=None):
    from ..biodivinit.PhyloGenerator import SimRandomPhylo as phyloGenerator
    tree_generator = phyloGenerator(n_species=n_species)
    # from ..biodivinit.PhyloGenerator import ReadRandomPhylo as phyloGenerator
    # phylo_obj = phyloGenerator(phylofolder="data_dependencies/phylo/", seed=seed)
    return tree_generator


class RunMode(Enum):
    # TODO: rename to FULLMONITORING, ONETIMEMONITORING etc.
    ORACLE = "ORACLE"
    STANDARD = "STANDARD"
    NOUPDATEOBS = "NOUPDATEOBS"
    PROTECTATONCE = "PROTECTATONCE"


# used in reinforce.RichProtectActionAdaptor
class BiodivEnvUtils(object):
    @staticmethod
    def getQuadrandCoord(grid_size, resolution):
        resolution_grid_size = grid_size / resolution
        x_coord = np.arange(0, grid_size + 1, resolution[0])
        y_coord = np.arange(0, grid_size + 1, resolution[1])
        quadrant_coords_list = []

        for x_i in np.arange(0, int(resolution_grid_size[0])):
            for y_i in np.arange(0, int(resolution_grid_size[1])):
                Xs = np.arange(x_coord[x_i], x_coord[x_i + 1])
                Ys = np.arange(y_coord[y_i], y_coord[y_i + 1])
                quadrant_coords_list.append([Xs, Ys])
        return quadrant_coords_list

    @staticmethod
    def getRichAction(action, grid_size, resolution):
        if action == 0:
            return Action(ActionType.Observe, 0, -1)
        # elif action == 1:
        #    return Action(ActionType.Observe, 0)
        else:
            cellList = BiodivEnvUtils.getQuadrandCoord(grid_size, resolution)
            return Action(ActionType.Protect, cellList[action - 1], action)

    @staticmethod
    def getRichProtectAction(action, grid_size, resolution):
        cellList = BiodivEnvUtils.getQuadrandCoord(grid_size, resolution)
        # print(grid_size, resolution, len(cellList), action)
        return Action(ActionType.Protect, cellList[action], action)


class BioDivEnv(gym.Env):
    """BioDiv Environment that follows gym interface"""

    metadata = {"render.modes": ["human_print", "human_plot", "dict_csv_ready"]}

    def __init__(self,
                 budget,
                 gridInitializer,
                 length=None,
                 n_species=None,
                 alpha=0.01,
                 K_max=None,
                 dispersal_rate=0.1,
                 disturbanceGenerator=None,
                 disturbance_sensitivity=None,
                 selectivedisturbanceInitializer=0,
                 selective_sensitivity=[],
                 max_fraction_protected=1,
                 immediate_capacity=False,
                 truncateToInt=False,
                 species_threshold=10,
                 rnd_alpha=0,
                 K_disturbance_coeff=1,
                 actions=[],
                 dispersal_before_death=0,
                 rnd_alpha_species=0,
                 climateModel=0,
                 ignoreFirstObs=0,
                 buffer_zone=1,
                 iterations=100,
                 verbose=True,
                 resolution=np.array([5, 5]),
                 numFeatures=10,
                 runMode=RunMode.STANDARD,
                 worker_id=0,
                 observeMode=1,
                 use_protection_cost=True,
                 rnd_sensitivities=0,
                 rnd_disturbance_init=-1,
                 tree_generator=0,
                 list_species_values=[],
                 species_carbon_value=None,
                 rewardMode="species",
                 climate_sensitivity=[],
                 climate_as_disturbance=1,
                 disturbance_dep_dispersal=0,
                 growth_rate=[1],
                 start_protecting=3,
                 species_risk_criteria=None,
                 update_previous_observation=True,
                 cost_pu=None,
                 habitat_suitability=None,
                 future_habitat_suitability=None,
                 max_protection_level=1,
                 dynamic_print=False,
                 precomputed_dispersal_probs=None,
                 use_small_grid=False,
                 K_species=None,
                 reward_min_protection=0,
                 reference_grid_pu=None,
                 feature_set=None,
                 ):
        super(BioDivEnv, self).__init__()

        if K_max is None or length is None or n_species is None:
            init_data = gridInitializer.getInitialState(1, 1, 1)
            K_max = np.einsum("xyz -> yz", init_data)[0][0]
            length = init_data.shape[1]
            n_species = init_data.shape[0]
            print("n_species, K_max, length", n_species, K_max, length)

        self._verbose = verbose
        self.lastActionType = None
        self.climateModel = climateModel
        self.climate_sensitivity = climate_sensitivity
        self.climate_as_disturbance = climate_as_disturbance
        self.rnd_alpha_species = rnd_alpha_species
        self.disturbance_dep_dispersal = disturbance_dep_dispersal
        self.actions = actions
        self.K_disturbance_coeff = K_disturbance_coeff
        self.rnd_alpha = rnd_alpha
        self.species_threshold = species_threshold
        self.truncateToInt = truncateToInt
        self.immediate_capacity = immediate_capacity
        self.selective_sensitivity = selective_sensitivity
        self.selectivedisturbanceInitializer = selectivedisturbanceInitializer
        self.disturbance_sensitivity = disturbance_sensitivity
        self.disturbanceGenerator = disturbanceGenerator
        self.dispersal_rate = dispersal_rate
        self.K_max = K_max

        self.alpha = alpha
        self.timeSinceLastObserve = None
        self.timeOfLastProtect = 0
        self.length = length
        self.num_quadrants = int((length / resolution[0]) * (length / resolution[1]))
        self.n_discrete_actions = (
            self.num_quadrants + 1
        )  # num_quadrants to Protect plus 1 to Observe (plus 1 to do nothing-removed)
        self.buffer_zone = buffer_zone  # size of buffer zone within protected area (with lower protection)
        self._max_protection_level = max_protection_level
        self.rnd_sensitivities = rnd_sensitivities
        self.rnd_disturbance_init = rnd_disturbance_init
        # calc absolute budget from a fraction
        if budget < 1:
            total_cost = Action.PROTECT_COST * (self.length ** 2)
            self._initialBudget = budget * total_cost
        else:
            self._initialBudget = budget
        self.action_space = spaces.Discrete(self.n_discrete_actions)
        self.num_features = numFeatures  # as we now have 7 features per quadrant TODO this needs to be in init
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(1, self.num_quadrants * self.num_features)
        )
        self.ignoreFirstObs = ignoreFirstObs

        self.iterations = iterations
        self.n_species = n_species
        self.resolution = resolution
        self.runMode = runMode
        self.observeMode = observeMode
        self.rewardMode = rewardMode
        self.reference_grid_pu = reference_grid_pu
        if self._verbose:
            print("init bioDivGrid.SimGrid")
        self.bioDivGrid = SimGrid(
            length,
            n_species,
            alpha,
            K_max,
            dispersal_rate,
            disturbanceGenerator,
            disturbance_sensitivity,
            selectivedisturbanceInitializer=selectivedisturbanceInitializer,
            selective_sensitivity=selective_sensitivity,
            immediate_capacity=immediate_capacity,
            truncateToInt=truncateToInt,
            species_threshold=species_threshold,
            rnd_alpha=rnd_alpha,
            K_disturbance_coeff=K_disturbance_coeff,
            dispersal_before_death=dispersal_before_death,
            actions=actions,
            rnd_alpha_species=rnd_alpha_species,
            climateModel=climateModel,
            climate_sensitivity=climate_sensitivity,
            climate_as_disturbance=self.climate_as_disturbance,
            disturbance_dep_dispersal=self.disturbance_dep_dispersal,
            growth_rate=growth_rate,
            habitat_suitability=habitat_suitability,
            future_habitat_suitability=future_habitat_suitability,
            precomputed_dispersal_probs=precomputed_dispersal_probs,
            K_species=K_species,
        )

        self._gridInitializer = gridInitializer
        if worker_id > 0:
            self._verbose = 0
        self._max_n_protected_cells = int(max_fraction_protected * self.length ** 2)
        self.protected_quadrants = []
        self.protection_sequence = []
        self.use_protection_cost = use_protection_cost
        self.cost_protected_quadrants = 0
        self.list_species_values = list_species_values
        self.list_species_values_init = list_species_values
        self.species_carbon_value = species_carbon_value
        self.tree_generator = tree_generator
        self._growth_rate = growth_rate
        self._baseline_cost = Action.PROTECT_COST * (
            self.resolution[0] * self.resolution[1]
        )
        self._cost_coeff = 0.4
        self._start_protecting = (
            start_protecting  # n. steps after which protection policy starts
        )
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self._default_action = Action(ActionType(4), 0, 0)
        self.update_previous_observation = update_previous_observation
        self._reward_weights = {'carbon': 10, 'species_risk': 10}

        # -- costs for empirical data
        self._protection_cost = cost_pu

        self.ext_risk_class = ExtinctionRisk # default class used for self.reset_risk_labels()
        self.species_risk_criteria = species_risk_criteria
        self.habitat_suitability = habitat_suitability
        self.future_habitat_suitability = future_habitat_suitability
        self.precomputed_dispersal_probs = precomputed_dispersal_probs
        self.calcReward = True
        self.dynamic_print = dynamic_print
        self.use_small_grid = use_small_grid
        self._K_species = K_species
        self._reward_min_protection = reward_min_protection
        self.feature_set = feature_set
        self.conv_block = None
        self._env_layers = None
        self.reset()

    def _initEnv(self):
        self.budget = self._initialBudget
        self.currentIteration = 0
        self.n_extant = self.n_species
        if self._verbose:
            print("re-loading grid...")

        if self.rnd_sensitivities:
            rr = random.randint(1000, 9999)
            rs = get_rnd_gen(rr)
            self.disturbance_sensitivity = np.zeros(self.n_species) + rs.random(self.n_species)
            self.selective_sensitivity = rs.beta(0.2, 0.7, self.n_species)
            self.climate_sensitivity = rs.beta(2, 2, self.n_species)
            # print(rr, "Rnd sens", disturbance_sensitivity[0:5])
        if self._verbose:
            print("Rnd sens", self.disturbance_sensitivity[0:5])
            print("Rnd climate sens", self.climate_sensitivity[0:5])
            print("RESOLUTION", self.resolution)

        if self.rnd_disturbance_init != -1:
            rr = random.randint(1000, 9999)
            distb_obj, selectivedistb_obj = get_disturbance(self.rnd_disturbance_init, seed=rr)
            self.disturbanceGenerator = distb_obj
            self.selectivedisturbanceInitializer = selectivedistb_obj

        if self.tree_generator:
            pass
        else:
            # random tree sampler
            if self._verbose:
                print("Generating tree...")
            self.tree_generator = get_phylo_generator(n_species=self.n_species)

        if self._verbose:
            print("Creating SimGrid...")
        self.bioDivGrid = SimGrid(
            self.length,
            self.n_species,
            self.alpha,
            self.K_max,
            self.dispersal_rate,
            self.disturbanceGenerator,
            self.disturbance_sensitivity,
            selectivedisturbanceInitializer=self.selectivedisturbanceInitializer,
            selective_sensitivity=self.selective_sensitivity,
            immediate_capacity=self.immediate_capacity,
            truncateToInt=self.truncateToInt,
            species_threshold=self.species_threshold,
            rnd_alpha=self.rnd_alpha,
            K_disturbance_coeff=self.K_disturbance_coeff,
            actions=self.actions,
            rnd_alpha_species=self.rnd_alpha_species,
            climateModel=self.climateModel,
            phyloGenerator=self.tree_generator,
            climate_sensitivity=self.climate_sensitivity,
            climate_as_disturbance=self.climate_as_disturbance,
            disturbance_dep_dispersal=self.disturbance_dep_dispersal,
            growth_rate=self._growth_rate,
            habitat_suitability=self.habitat_suitability,
            future_habitat_suitability=self.future_habitat_suitability,
            precomputed_dispersal_probs=self.precomputed_dispersal_probs,
            K_species=self._K_species,
        )
        if self._verbose:
            print("Creating initGrid...")

        self.bioDivGrid.initGrid(self._gridInitializer)

        self.quadrant_coords_list = BiodivEnvUtils.getQuadrandCoord(
            self.length, self.resolution
        )

        sp_coord = self.bioDivGrid.get_species_mid_coordinate()
        max_value_coord = []
        if len(self.list_species_values_init) == 0 or self.rnd_sensitivities:
            randval = np.random.choice([0.1, 10], self.n_species, p=[0.8, 0.2])
            self.list_species_values = randval / np.sum(randval) * self.n_species
            max_value_coord = np.concatenate( (np.random.choice(range(self.length), 2),
                                               np.exp(np.random.uniform(np.log(0.2), np.log(3), 1)) ))

        elif len(self.list_species_values_init) == 2:
            max_value_coord = self.list_species_values_init[0]
            self.list_species_values = self.list_species_values_init[1]
        else:
            self.list_species_values = self.list_species_values_init + 0

        if len(max_value_coord):
            # small rnd_exponent: flatter, large rnd_exponent: sharper
            rnd_exponent = max_value_coord[2] # np.exp(np.random.uniform(np.log(0.2), np.log(3)))
            sp_dist_from_opt_lat = abs(max_value_coord[0] - sp_coord[0, :])
            sp_dist_from_opt_lon = abs(max_value_coord[1] - sp_coord[1, :])
            sp_dist_from_opt = .5 + np.sqrt(
                sp_dist_from_opt_lat ** 2 + sp_dist_from_opt_lon ** 2
                ) ** rnd_exponent
            geo_values = []
            for i in range(len(self.list_species_values)):
                geo_values.append(
                    self.list_species_values[i] / (sp_dist_from_opt[i] ** 2)
                )

            geo_values = np.array(geo_values) / np.sum(geo_values) * self.n_species
            self.list_species_values = geo_values + 0

             # for i in range(len(self.list_species_values)):
            #     print(i, sp_dist_from_opt[i], np.round(self.list_species_values[i], 4), np.round(geo_values[i], 4))
        if self.species_carbon_value is None:
            # logistic
            tmp = np.log(self.list_species_values + 1)
            self.species_carbon_value = 1 / (1 + np.exp(- 10 * (tmp - np.mean(tmp))))

        if DEBUG:
            print("Rnd sp values: ", self.list_species_values[0:5])
            print("Rnd carb values: ", self.species_carbon_value[0:5], rnd_exponent,
                  np.sum(self.species_carbon_value), np.sum(self.bioDivGrid.getCarbonValue_cell()),
                  np.sum(self.grid_obj_previous.getCarbonValue_cell()))

        self.bioDivGrid.set_species_carbon_value(self.species_carbon_value)

        self.bioDivGrid.set_reference_grid_pu(self.reference_grid_pu)
        self._init_total_carbon = np.sum(self.bioDivGrid.getCarbonValue_cell())
        self._init_total_population = np.sum(self.bioDivGrid.h)

        self.current_carbon = np.sum(self.bioDivGrid.getCarbonValue_cell())
        # set species value in biodivGrid for reference
        self.bioDivGrid.set_species_values(self.list_species_values)

        self.value_extant_sp = np.sum(self.list_species_values)
        self.pd_extant_sp = self.bioDivGrid.totalPDextantSpecies()
        self.n_protected_cells = np.sum(self.bioDivGrid.protection_matrix > 0.0)

        self.n_extant_init = np.copy(self.n_extant)
        self.value_extant_sp_init = np.copy(self.value_extant_sp)
        self.pd_extant_sp_init = np.copy(self.pd_extant_sp)
        self.done_protect_all_steps = 0

        if self.species_risk_criteria is None:
            if self._verbose:
                print("Setting up species_risk_criteria")
            self.species_risk_criteria = self.ext_risk_class(self.bioDivGrid)

        self._init_sp_ext_risk = self.getExtinction_risk_labels()

        self.current_risk = self.risk_label_counts(normalize=True)
        # create rnd species ID and grid of indices to map cells to quadrants
        l, i = get_quadrant_indx_grid(self.length, self.resolution), np.arange(self.num_quadrants)
        sp_rnd_id = np.sort(np.random.random(self.n_species))
        # broadcast 's' + 'xy' = 'sxy'
        l_sp = sp_rnd_id[:, np.newaxis, np.newaxis] + l[np.newaxis, :, :]
        # broadcast 's' + 'q' = 'sq' (where q = x * y, number of quadrants)
        i_sp = sp_rnd_id[:, np.newaxis] + i[np.newaxis, :]
        self._quandrant_grid_indx = [l, i, l_sp, i_sp]
        self._total_natural_carbon = None

        self._baseline_cost_quadrant = self.getProtectBaselineCostQuadrant()

        if self._verbose:
            print("Copying grid...")

        if self.use_small_grid:
            self.grid_obj_previous = smallGrid(self.bioDivGrid)
            self.grid_obj_most_recent = smallGrid(self.bioDivGrid, env_layers=self._env_layers)
        else:
            self.grid_obj_previous = copy.deepcopy(self.bioDivGrid)
            self.grid_obj_most_recent = copy.deepcopy(self.bioDivGrid)



        """[info["ExtantSpecies"], info["ExtantSpeciesValue"],
             info["ExtantSpeciesPD"], info["TotalCarbon"]
             ] + list(self.risk_label_counts(normalize=True)) + self.get_metrics()"""


        self.history_var_names = np.array(['extant_sp', 'value','PD','carbon', 'population',
                                           'CR', 'EN', 'VU', 'NT', 'LC',
                                           'MSA', 'cost', 'protected_species',
                                           'net_carbon', 'carbon_in_protected_area'])

        info = self._getInfo()
        self.history = [[info["ExtantSpecies"], info["ExtantSpeciesValue"],
             info["ExtantSpeciesPD"], info["TotalCarbon"],
             info["TotPopulation"]] + list(self.risk_label_counts(normalize=False)) + self.get_metrics() + [
            self._initialBudget - self.budget,
            self.n_species - info['non_protected_species'],
            np.sum(self.bioDivGrid.getCarbonValue_cell()) - self._init_total_carbon,
            np.sum(self.bioDivGrid.getCarbonValue_cell()[self.bioDivGrid.protection_matrix > 0])

        ]]
            # [[1, 1, 1, 1, 1] + list(self.risk_label_counts(normalize=True)) + self.get_metrics()]

        self.current_protected_range_fraction = np.sum(
            self.bioDivGrid.protectedRangePerSpecies() / self.bioDivGrid.geoRangePerSpecies())



    def init_costs_budget(self, budget):
        cost_array = self._protection_cost + 0
        # add a minimum cost per unit and rescale
        if np.min(cost_array) == 0:
            if np.max(cost_array) == 0:
                min_cost = 1
            else:
                min_cost = 0.01 * np.min(
                    cost_array[cost_array > 0]
                )  # 1% of the cheapest cell with a cost
            cost_array[cost_array == 0] = min_cost

        # if not ignore_pu_status:
        #     status_array = np.array(cost_tbl["status"])[cost_tbl["id"].isin(emp._pus_id)]
        #     p = status_array.reshape(emp._init_protection_matrix.shape)
        #     emp.reset_init_protection_matrix(p)

        # rescale cost
        # if we want the budget to be a fraction:
        if budget <= 1:
            cost_array = cost_array / np.mean(cost_array)
            # total_cost = np.sum(cost_array)
            # set a budget sufficient to protect 10% of cheapest PUs
            budget = budget * (np.min(cost_array) * self.bioDivGrid.h.size)
        # else the budget will be in dollars or whatever unit the manager wants
        self.budget = budget

    def risk_label_counts(self, normalize=False):
        risk_labels = self.getExtinction_risk_labels()
        labels, label_count = np.unique(risk_labels, return_counts=True)
        for i in self.species_risk_criteria.available_labels:
            if i not in labels:
                label_count = np.insert(label_count, i, 0)

        if normalize:
            label_count = label_count / np.sum(label_count)
        return label_count


    def _protectCellList(self, cellList):
        """from an action in the action space to an actual protection matrix and selective disturbance matrix
        currently not updating the selective disturbance
        """
        # TODO: do we need this np.copy?
        protectionMatrix = np.copy(self.bioDivGrid.protection_matrix)
        pcellList = []
        for i in cellList[0]:
            for j in cellList[1]:
                if self.buffer_zone > 0:
                    if (
                        i in cellList[0][: self.buffer_zone]
                        or i in cellList[0][-self.buffer_zone :]
                        or j in cellList[1][: self.buffer_zone]
                        or j in cellList[1][-self.buffer_zone :]
                    ):
                        protectionMatrix[i][j] = 0.5
                    else:
                        protectionMatrix[i][j] = self._max_protection_level
                else:
                    protectionMatrix[i][j] = self._max_protection_level
                pcellList.append((i, j))
        # print(f'Protecting Cells: {pcellList}')
        self.bioDivGrid.setProtectionMatrix(protectionMatrix)

    def _canProtect(self):
        canProtect = (
            self.bioDivGrid.protection_matrix > 0
        ).sum() < self._max_n_protected_cells
        return canProtect

    def get_protected_fraction(self):
        b_tmp = self.bioDivGrid.geoRangePerSpecies()
        protected_range_fraction = np.zeros(self.n_species)
        protected_range_fraction[b_tmp > 0] = self.bioDivGrid.protectedRangePerSpecies()[b_tmp > 0] / b_tmp[b_tmp > 0]
        return protected_range_fraction

    def observe(self, timeSinceLastObserve=0.0):
        self.timeSinceLastObserve = timeSinceLastObserve
        if self.use_small_grid:
            # ALTERNATIVE COPY
            if self.update_previous_observation:
                self.grid_obj_previous = smallGrid(self.grid_obj_most_recent)
            self.grid_obj_most_recent = smallGrid(self.bioDivGrid)
        else:
            if self.update_previous_observation:
                self.grid_obj_previous = copy.deepcopy(self.grid_obj_most_recent)
            self.grid_obj_most_recent = copy.deepcopy(self.bioDivGrid)

        # return extract_features(self.grid_obj_most_recent, self.grid_obj_previous,
        #                         quadrant_resolution = self.resolution,
        #                         current_protection_matrix = self.bioDivGrid.protection_matrix,
        #                         mode=self.observeMode,
        #                         cost_quadrant = self._baseline_cost + self.getProtectCostQuadrant(),
        #                         budget=self.budget,
        #                         sp_values=self.list_species_values)

    def update_protected_quadrants_in_lastObs(self):
        sys.exit("update_protected_quadrants_in_lastObs not implemented")
        # Updates features to account for the latest protected quadrant
        # return extract_features(self.grid_obj_most_recent, self.grid_obj_previous,
        #                         quadrant_resolution = self.resolution,
        #                         current_protection_matrix = self.bioDivGrid.protection_matrix,
        #                         mode=self.observeMode,
        #                         cost_quadrant = self._baseline_cost + self.getProtectCostQuadrant(),
        #                         budget=self.budget,
        #                         sp_values=self.list_species_values)

    def getProtectCostQuadrant(self, coordinates=[], fun=np.sum):
        if self._protection_cost is None:
            # with disturbance = 1, protection quadrants = 5x5, coeff_cost = 0.2, baseline_cost = 5
            # price doubles = 10
            # with disturbance = 1, protection quadrants = 5x5, coeff_cost = 0.4, baseline_cost = 5
            # price triples = 15 = 5 + (5*5)*0.4
            # set cost of already protected areas to 0
            dist_tmp = self.bioDivGrid.disturbance_matrix * (1 - self.bioDivGrid.protection_matrix)
        else:
            "Empirical cost + effect of disturbance"
            dist_tmp = self._protection_cost * (1 - self.bioDivGrid.protection_matrix)

        if len(coordinates) == 0:
            if self.use_protection_cost:
                c_tmp = ndimage.sum(dist_tmp, labels=self._quandrant_grid_indx[0], index=self._quandrant_grid_indx[1])
                return self._cost_coeff * c_tmp
            else:
                return []
        else:
            # print("coordinates", coordinates)
            if self.use_protection_cost:
                try:
                    # calculate for one quadrant
                    quadrant_coords = np.meshgrid(coordinates[0], coordinates[1])
                    return self._cost_coeff * fun(dist_tmp[tuple(quadrant_coords)])
                except:
                    return self._cost_coeff * fun(dist_tmp.flatten()[coordinates])
            else:
                return 0


    def getProtectBaselineCostQuadrant(self, coordinates=[]):
        cost_tmp = (self.bioDivGrid._K_max > 0) * self._baseline_cost

        if len(coordinates) == 0:
            if self.use_protection_cost:
                c_tmp = ndimage.mean(cost_tmp, labels=self._quandrant_grid_indx[0], index=self._quandrant_grid_indx[1])
                return c_tmp
            else:
                return []
        else:
            if self.use_protection_cost:
                # calculate for one quadrant
                quadrant_coords = np.meshgrid(coordinates[0], coordinates[1])
                return np.mean(cost_tmp[tuple(quadrant_coords)])
            else:
                return 0


    def set_reward_min_protection(self, r):
        self._reward_min_protection = r

    def set_max_n_protected_cells(self, r):
        self._max_n_protected_cells = r

    def set_env_layers(self, s):
        self._env_layers = s

    # def getCarbonValue_cell(self):
    #     return np.einsum('sxy,s -> xy', self.bioDivGrid.h, self.species_carbon_value)
    #
    # def getCarbonQuadrant(self):
    #     return ndimage.sum(self.getCarbonValue_cell(),
    #                        labels=self._quandrant_grid_indx[0],
    #                        index=self._quandrant_grid_indx[1])
    #
    # def getCarbonQuadrantPotential(self):

    def getExtinction_risk_labels(self, grid=None):
        if grid is None:
            return self.species_risk_criteria.classify_species(state = self.bioDivGrid)
        else:
            return self.species_risk_criteria.classify_species(grid)

    def getFutureExtinction_risk_labels(self, grid=None):
        if grid is None:
            return self.species_risk_criteria.predict_future_species(self.bioDivGrid)
        else:
            return self.species_risk_criteria.predict_future_species(grid)

    def calc_connectivity_reward(self, verbose=False):
        if self.conv_block is None:
            self.conv_block = np.ones((3,3))

        tmp = self.bioDivGrid.protection_matrix.flatten(
        )[:-(self.bioDivGrid.protection_matrix.size - self.bioDivGrid._n_pus)] + 0
        m_grid = np.zeros(self.bioDivGrid._reference_grid_pu.shape)
        m_grid[self.bioDivGrid._reference_grid_pu > 0] += tmp

        c = ndimage.convolve(m_grid, self.conv_block, mode='constant', origin=0)
        if verbose:
            print(c)
        c[c == 1] = 0
        # the reward is relative to the number of protected cells
        return np.sum(c) / np.sum(self.bioDivGrid.protection_matrix)

    def step(self, action: Action = None,
             skip_env_step: bool = False, skip_dispersal=False, update_suitability=False):
        if action is None:
            action = self._default_action
        # TODO: check/fix TIMETOPROTECT
        TIMETOPROTECT = self._start_protecting
        did_protect = 0
        # if self._verbose:
        #     print(self.lastObs.stats_quadrant[0:5,:])
        # this returns an observation, the reward, a flag to indicate the end of the experiment and additions info in a dict
        # execute action and pay cost of the action
        self.lastActionType = action.actionType

        if self.runMode == RunMode.ORACLE:
            # only allowed action is protect and do observe at no cost
            # if action.actionType != ActionType.Protect:
            #     raise Exception("only allowed action is protect in ORACLE mode")
            if (
                self.currentIteration < TIMETOPROTECT
            ):  # before step 3 some features are not available
                action.actionType = ActionType.NoAction
            self.observe()
            # self.lastObs = self.observe()

        if self.runMode == RunMode.PROTECTATONCE:
            skip_env_step = True
            # only allowed action is protect and do observe at no cost
            if action.actionType != ActionType.Protect:
                raise Exception("only allowed action is protect in PROTECTATONCE mode")
            if (
                self.currentIteration < TIMETOPROTECT
            ):  # before step 3 some features are not available
                action.actionType = ActionType.NoAction
                # self.lastObs = self.observe() # only update biodiv if a step was made

        if self.runMode == RunMode.NOUPDATEOBS:
            # only allowed action is protect and do observe at no cost
            if action.actionType != ActionType.Protect:
                raise Exception("only allowed action is protect in NOUPDATEOBS mode")
            if (
                self.currentIteration < TIMETOPROTECT
            ):  # before step 3 some features are not available
                action.actionType = ActionType.NoAction

        if action.actionType == ActionType.Observe:
            if self.budget >= Action.OBSERVE_COST:
                self.timeSinceLastObserve = 0.0
                self.observe()
                self.budget -= Action.OBSERVE_COST

        elif action.actionType == ActionType.Protect:
            added_protection_cost = self.getProtectCostQuadrant(coordinates=action.value)
            # print("action.value", action.value, added_protection_cost + self._baseline_cost_quadrant[action.value_quadrant])
            # print(action.value_quadrant, Action.PROTECT_COST + added_protection_cost, self.budget)
            # print(self.protected_quadrants)
            # print("action.value_quadrant", action.value_quadrant, "added_protection_cost", added_protection_cost)
            cost = np.sum(self._baseline_cost_quadrant[np.array(action.value_quadrant)]) + np.sum(added_protection_cost)
            # print(cost, self.budget)
            if self.budget >= cost:
                if self._canProtect():
                    # do not observe the state, keep knowledge as last step, update protection matrix
                    try:
                        self._protectCellList(action.value)
                    except:
                        protectionMatrix = np.copy(self.bioDivGrid.protection_matrix).flatten()
                        protectionMatrix[action.value] = self._max_protection_level
                        self.bioDivGrid.setProtectionMatrix(protectionMatrix.reshape(self.bioDivGrid.protection_matrix.shape))

                    self.protected_quadrants.append(action.value_quadrant)
                    self.budget -= cost #self._baseline_cost_quadrant[action.value_quadrant] + added_protection_cost
                    # self.lastObs = self.update_protected_quadrants_in_lastObs()
                    self.timeOfLastProtect = self.currentIteration + 0
                    if self.cost_protected_quadrants == 0: # store average cost of protection
                        self.cost_protected_quadrants = (
                            self._baseline_cost + added_protection_cost
                        )
                    else:
                        self.cost_protected_quadrants = (self.cost_protected_quadrants + (self._baseline_cost + added_protection_cost))/2
                    did_protect = 1
                    self.protection_sequence.append(action.value)

        elif action.actionType == ActionType.NoObserve:
            if self.budget >= Action.NOOBSERVE_COST:
                # self.lastObs = extract_features(self.grid_obj_most_recent, self.grid_obj_previous,
                #                                 quadrant_resolution = self.resolution,
                #                                 current_protection_matrix = self.bioDivGrid.protection_matrix,
                #                                 cost_quadrant = self._baseline_cost + self.getProtectCostQuadrant(),
                #                                 budget=self.budget)
                self.budget -= Action.NOOBSERVE_COST
            # print(self.grid_obj_previous.geoRangePerSpecies())

            # do nothing
            pass
        elif action.actionType == ActionType.NoAction:
            # do nothing
            pass

        else:
            raise NotImplemented("not yet implemented!!")
        # Execute one time step within the environment
        if self._verbose:
            if self.currentIteration == 0:
                self.tmp = 0
            d1 = np.round(np.mean(self.bioDivGrid._disturbance_matrix), 2)
            d2 = np.round(np.mean(self.bioDivGrid._selective_disturbance_matrix), 2)
            # TODO: improve screen output
            s = f"Time step: {self.bioDivGrid._counter} (step: {self.currentIteration})"
            if self.runMode == RunMode.PROTECTATONCE:
                if did_protect == 1:
                    self.tmp += 1
                    s = f"  PU: {1 + self.currentIteration - self._start_protecting}"
                elif 1 + self.currentIteration > self._start_protecting:
                    s = f"Step: {1 + self.currentIteration - self.tmp}"

            screen_out = s + \
                f" N. protected cells: {np.sum(self.bioDivGrid.protection_matrix > 0.)}" + \
                f" Budget: {np.round(self.budget,2)}" + \
                f" N. species: {self.bioDivGrid.numberOfSpecies()}" + \
                f" Population: {np.round(np.log10(np.sum(self.bioDivGrid.h)), 2)}" + \
                f" Disturbance: {np.round(d1,2)}, {np.round(d2,2)}"
                # f" Carbon: {np.round(np.log10(1 + self.current_carbon), 2)}" + \

            if self.dynamic_print:
                print_update(screen_out)
            else:
                print(screen_out)


        if not skip_env_step: # self.runMode != RunMode.PROTECTATONCE
            if DEBUG:
                print("step", self.bioDivGrid._counter, self.currentIteration, self.iterations)
            self.bioDivGrid.step(skip_dispersal=skip_dispersal, update_suitability=update_suitability)
        else:
            # print("\nskipping step", self.bioDivGrid._counter, self.currentIteration)
            pass

        # if did_protect == 0:  # finished budget, continue with simulation
        #     self.observe()  # only update biodiv if a step was made
        #     if self.done_protect_all_steps == 1:
        #         self.bioDivGrid.step(fast_dist=False)
        #         self.done_protect_all_steps = 0
        #     else:
        #         self.bioDivGrid.step(fast_dist=True)
        # else:
        #     self.done_protect_all_steps = 1

        # build output by stacking obs and time till last obs
        richObs = self._enrichObs()

        # update counters, compute reward and done flag
        self.currentIteration += 1
        self.timeSinceLastObserve += 1

        if self.calcReward is False:
            reward = 0
            reward_c = 0
            reward_sp = 0
            reward_cost = 0
            reward_lab = 0
            reward_protect = 0
            protected_range_fraction = self.current_protected_range_fraction + 0
        elif self.rewardMode == "species":  # use species loss
            reward = self.bioDivGrid.numberOfSpecies() - self.n_extant
        elif self.rewardMode == "value":  # use sp value
            reward = (
                np.sum(self.list_species_values[self.bioDivGrid.extantSpeciesID()])
                - self.value_extant_sp
            )
        elif self.rewardMode == "area":  # amount of protected area
            reward = (
                np.sum(self.bioDivGrid.protection_matrix > 0.0) - self.n_protected_cells
            ) / (self.resolution[0] * self.resolution[1])
        elif self.rewardMode == "pd":  # use PD loss
            reward = self.bioDivGrid.totalPDextantSpecies() - self.pd_extant_sp
        elif self.rewardMode == "carbon" or self.rewardMode == "future_carbon":  # net carbon (%)
            if self._total_natural_carbon is None:
                self.set_total_natural_carbon(np.sum(self.grid_obj_previous.getCarbonValue_cell()))

            tot_reward = np.sum(self.bioDivGrid.getCarbonValue_cell()) / self._total_natural_carbon * 100
            reward = tot_reward - self._cumrewards[0]

            #
            #
            # reward = ((np.sum(self.bioDivGrid.getCarbonValue_cell())
            #            - self.current_carbon) / self._total_natural_carbon * 100)
            # self._init_total_carbon) * self._reward_weights['carbon']
            # (self._total_natural_carbon - self._init_total_carbon)
        elif self.rewardMode == "ext_risk":
            reward = np.sum((self.risk_label_counts(normalize=True) *
                             self.species_risk_criteria.risk_weights) -
                      (self.current_risk *
                       self.species_risk_criteria.risk_weights)) * self._reward_weights['species_risk']
            # print("step reward:", reward)

        elif self.rewardMode == "ext_risk_carbon":
            reward_sp = np.sum((self.risk_label_counts(normalize=True) *
                             self.species_risk_criteria.risk_weights) -
                      (self.current_risk *
                       self.species_risk_criteria.risk_weights)) * self._reward_weights['species_risk']
            reward_c = ((np.sum(self.bioDivGrid.getCarbonValue_cell())
                              - self.current_carbon) /
                             self._init_total_carbon) * self._reward_weights['carbon']
            if DEBUG:
                print("REWARDS:", np.array([reward_sp, reward_c]), self._reward_weights)
                print(self.risk_label_counts(normalize=False), self.n_species * self.current_risk)
            reward = reward_sp + reward_c
        elif self.rewardMode == "ext_risk_protect" or  self.rewardMode == "star_t":
            reward_lab = np.sum((self.risk_label_counts(normalize=True) *
                             self.species_risk_criteria.risk_weights) -
                      (self.current_risk *
                       self.species_risk_criteria.risk_weights)) * self._reward_weights['species_risk']

            protected_range_fraction = np.sum(self.get_protected_fraction())
            reward_protect = protected_range_fraction - self.current_protected_range_fraction
            reward = reward_lab + reward_protect
        elif self.rewardMode == "sp_risk_protect" or self.rewardMode == "sp_risk_protect_future":
            rl = self.getExtinction_risk_labels()

            protected_area_per_species = self.bioDivGrid.protectedRangePerSpecies()
            if self._reward_min_protection > 0:
                n_non_protected_species = np.sum(protected_area_per_species < self._reward_min_protection)
                reward_leftout_species = - n_non_protected_species / self.n_species * 100
            else:
                reward_leftout_species = 0
            # REWARD
            # weighted by class
            geo_range = self.bioDivGrid.geoRangePerSpecies()
            geo_range[geo_range < 1] = 1
            # fraction of non protected range
            pr_fr = 1 - (protected_area_per_species / geo_range)

            reward_protected_species = np.sum(pr_fr * (self.species_risk_criteria.risk_weights[rl]))

            reward_species = (reward_protected_species) / self.n_species * 100
            reward_disturbance = np.sum(
                self.bioDivGrid.disturbance_matrix * (self.bioDivGrid.protection_matrix)
            ) / np.sum(self.bioDivGrid.disturbance_matrix) * 100


            tot_reward = reward_species + reward_disturbance + reward_leftout_species
            # tot_reward = -np.sum(non_protected_species_id)
            reward = tot_reward - self._cumrewards[0]
            # if self._verbose:
            #     print("\nReward internal:", reward, reward_protected_species,
            #           tot_reward, reward_disturbance, "reward_leftout_species",reward_leftout_species,
            #           "n_non_protected_species", n_non_protected_species,
            #           self._cumrewards)
            # # print(rl)
            # print(pr_fr)
        elif self.rewardMode in ["sp_risk_protect_pop", "sp_risk_protect_pop_future", "sp_risk_conv"]:
            rl = self.getExtinction_risk_labels()

            protected_pop_per_species = self.bioDivGrid.protectedIndPerSpecies()
            n_non_protected_species = len(protected_pop_per_species[protected_pop_per_species < 1])
            # REWARD
            # weighted by class
            pop_size = self.bioDivGrid.individualsPerSpecies()
            # fraction of non protected range
            # TODO: could replace pop_size with self.grid_obj_previous.individualsPerSpecies() ?
            #  i.e. fraction of natural state
            pr_fr = np.zeros(self.n_species)
            # treat extinct as lowest category (CR), with no protection
            pr_fr[pop_size > 0] = 1 - (protected_pop_per_species[pop_size > 0] / pop_size[pop_size > 0])

            # rl[pop_size == 0] = 0


            #--- NEW REWARD
            avg_protection_rl_class = np.ones(self.species_risk_criteria.n_labels)
            for i in range(self.species_risk_criteria.n_labels):
                if i in rl:
                    avg_protection_rl_class[i] = 1 - np.mean(pr_fr[rl == i])
                # elif i == 4:  # no LC in the dataset
                #     avg_protection_rl_class.append(0)
                # else:  # no other classes in the dataset
                #     avg_protection_rl_class.append(1)

            reward_species = np.sum(avg_protection_rl_class * -self.species_risk_criteria.risk_weights)
            reward_species = reward_species / np.sum(-self.species_risk_criteria.risk_weights) * 100
            #----

            # reward_protected_species = np.sum(pr_fr * (self.species_risk_criteria.risk_weights[rl]))
            #
            # reward_species = (reward_protected_species) / self.n_species * 100
            # reward_disturbance = np.sum(
            #     self.bioDivGrid.disturbance_matrix * (self.bioDivGrid.protection_matrix)
            # ) / np.sum(self.bioDivGrid.disturbance_matrix) * 100
            reward_disturbance = 0


            tot_reward = reward_species + reward_disturbance
            # tot_reward = -np.sum(non_protected_species_id)
            reward = tot_reward - self._cumrewards[0]
            # if self._verbose:
            #     print("\nReward internal:", reward, reward_species,
            #           tot_reward, reward_disturbance,
            #           "n_non_protected_species", n_non_protected_species,
            #           np.sum(self.bioDivGrid._protection_matrix),
            #
            #           )
            #     print(protected_pop_per_species[:10])
            #     print(pr_fr[:10])

        elif self.rewardMode == "pareto_mlt":
            # carbon reward
            if self._total_natural_carbon is None:
                self.set_total_natural_carbon(np.sum(self.grid_obj_previous.getCarbonValue_cell()))
            if self.currentIteration == 1:
                # print("\nself._init_total_carbon = np.sum(self.bioDivGrid.getCarbonValue_cell())",
                #       np.sum(self.bioDivGrid.getCarbonValue_cell()))
                self._init_total_carbon = np.sum(self.bioDivGrid.getCarbonValue_cell())

            if 'carbon' in self._reward_weights.keys():
                tot_reward = np.sum(self.bioDivGrid.getCarbonValue_cell()) - self._init_total_carbon
                tot_reward = tot_reward / (self._total_natural_carbon - self._init_total_carbon) * 100
                reward_c = (tot_reward * self._reward_weights['carbon'] - self._cumrewards['carbon'])
                self._step_rewards['carbon'] = float(reward_c)


            if 'population' in self._reward_weights.keys():

                tot_reward = self.bioDivGrid.protectedIndPerSpecies() / self.bioDivGrid.individualsPerSpecies()
                pop_scaler = 0.5 # values < 1 give more importance to first portion of protected population
                # e.g. np.linspace(0, 1, 11) ** 0.5
                tot_reward = np.mean(tot_reward ** pop_scaler) * 100
                reward_c = (tot_reward * self._reward_weights['population'] - self._cumrewards['population'])
                self._step_rewards['population'] = float(reward_c)


            # species reward
            if 'species_risk' in self._reward_weights.keys():
                rl = self.getExtinction_risk_labels()
                protected_pop_per_species = self.bioDivGrid.protectedIndPerSpecies()
                pop_size = self.bioDivGrid.individualsPerSpecies()
                pr_fr = np.zeros(self.n_species)
                pr_fr[pop_size > 0] = 1 - (protected_pop_per_species[pop_size > 0] / pop_size[pop_size > 0])
                avg_protection_rl_class = np.ones(self.species_risk_criteria.n_labels)
                for i in range(self.species_risk_criteria.n_labels):
                    if i in rl:
                        avg_protection_rl_class[i] = 1 - np.mean(pr_fr[rl == i])
                tot_reward_species = np.sum(avg_protection_rl_class * -self.species_risk_criteria.risk_weights)
                tot_reward_species = tot_reward_species / np.sum(-self.species_risk_criteria.risk_weights) * 100
                reward_sp = (tot_reward_species * self._reward_weights['species_risk'] - self._cumrewards['species_risk'])
                self._step_rewards['species_risk'] = float(reward_sp)

            if 'env_distance' in self._reward_weights.keys():
                if self._env_layers is None:
                    sys.exit("_env_layers not provided.\n")

                else:
                    if self.previous_protection_matrix is None:
                        self.previous_protection_matrix = np.zeros(self.bioDivGrid.protection_matrix)

                    sp_rewards = np.zeros(self.bioDivGrid._n_species)
                    for species_i in range(self.bioDivGrid._n_species):
                        env_sp_protect = self._env_layers[:,
                                         (self.bioDivGrid.protection_matrix * self.bioDivGrid.h[species_i]) > 0].reshape(
                            (self._env_layers.shape[0], np.sum((self.bioDivGrid.protection_matrix * self.bioDivGrid.h[species_i]) > 0)))

                        delta_protection_m = self.bioDivGrid.protection_matrix - self.previous_protection_matrix
                        # shape = (env_layers, added_protected_cells x occurrence of species_i)
                        env_sp_new_protect = self._env_layers[:, (delta_protection_m * self.bioDivGrid.h[species_i]) > 0].reshape(
                            (self._env_layers.shape[0], np.sum((delta_protection_m * self.bioDivGrid.h[species_i]) > 0)))

                        # env difference: shape (n_env_layers, n_protected_cells, n_added_protected_cells)
                        env_diff = env_sp_new_protect[:, np.newaxis, :] - env_sp_protect[:, :, np.newaxis]

                        reward = 0
                        if np.sum(self.bioDivGrid.protection_matrix) > 0:
                            env_diff_init = env_sp_new_protect[:, np.newaxis, :] - env_sp_new_protect[:, :, np.newaxis]
                            # if no cells are already protected
                            eucl_diff = np.sqrt(np.sum(env_diff_init ** 2, axis=0))
                            eucl_diff[np.diag_indices(eucl_diff.shape[0])] = np.nan
                            if eucl_diff.size > 1:
                                reward = np.mean(np.nanmin(eucl_diff, 1))
                            print("Reward 1: ", reward)

                        if np.sum(env_diff) > 0:
                            # shape (n_protected_cells, n_added_protected_cells)
                            eucl_diff = np.sqrt(np.sum(env_diff ** 2, axis=0))
                            # minimum distance to a protected area calculated for each additional area
                            min_diff = np.min(eucl_diff, axis=0)
                            print("Reward 2: ", np.mean(min_diff))
                            reward += np.mean(min_diff)

                        sp_rewards[species_i] = reward

                    self._step_rewards['env_distance'] = float(np.sum(sp_rewards))
                    self.previous_protection_matrix = self.bioDivGrid.protection_matrix + 0

            if 'connectivity' in self._reward_weights.keys():
                tot_reward = self.calc_connectivity_reward()
                reward_c = (tot_reward * self._reward_weights['connectivity'] - self._cumrewards['connectivity'])
                self._step_rewards['connectivity'] = float(reward_c)



            #---
            # rew = np.sum(self.risk_label_counts(normalize=False) * np.array([0, 0.5, 0.707, 0.866, 1]))
            # rew_norm = rew / self.n_species * 100
            # reward_sp = (rew_norm * self._reward_weights['species_risk'] - self._cumrewards[1])
            #---

            # print("\n reward_c: %s  reward_sp: %s  reward_cost: %s %s" % (reward_c, reward_sp, reward_cost, tot_reward_cost))
            # print(self.budget , self._initialBudget)
            # print("\nC:", self._init_total_carbon, self._total_natural_carbon, np.sum(self.bioDivGrid.getCarbonValue_cell()) )
            # reward cost
            if 'cost' in self._reward_weights.keys():
                tot_reward_cost = self.budget / self._initialBudget * 100
                reward_cost = (tot_reward_cost * self._reward_weights['cost'] - self._cumrewards['cost'])
                self._step_rewards['cost'] = float(reward_cost)

            reward = sum(self._step_rewards.values())
            # print("\n\nstep_rewards", self._step_rewards, reward)

        else:
            sys.exit("\nrewardMode not defined!\n")

        self.n_extant = self.bioDivGrid.numberOfSpecies()
        self.value_extant_sp = np.sum(
            self.list_species_values[self.bioDivGrid.extantSpeciesID()]
        )
        self.pd_extant_sp = self.bioDivGrid.totalPDextantSpecies()
        self.n_protected_cells = np.sum(self.bioDivGrid.protection_matrix > 0.0)
        self.current_carbon = np.sum(self.bioDivGrid.getCarbonValue_cell())
        self.current_risk = self.risk_label_counts(normalize=True) + 0
        if self.rewardMode == "ext_risk_protect":
            self.current_protected_range_fraction = protected_range_fraction
        # flag it done when it reaches the # of iterations
        done = self.currentIteration ==  self.iterations
        # print("DONE: self.currentIteration ==  self.iterations", self.currentIteration, self.iterations)
        # done = self.bioDivGrid._counter == self.iterations
        info = self._getInfo()

        # try:
        #     print("\nReward:", reward, reward_lab, reward_protect, self._cumrewards)
        #     print(self.risk_label_counts(normalize=True))
        #     print(self.current_risk)
        # except:
        #     pass

        if self.rewardMode == "ext_risk_carbon":
            self._cumrewards += np.array([reward_c, reward_sp])
            info['reward_c'] = self._cumrewards[0]
            info['reward_sp'] = self._cumrewards[1]
        elif self.rewardMode == "pareto_mlt":
            # self._cumrewards += np.array([reward_c, reward_sp, reward_cost])
            self._cumrewards = {key: self._cumrewards.get(key, 0) + self._step_rewards.get(key, 0) for key in set(self._cumrewards) | set(self._step_rewards)}

            for k in self._cumrewards.keys():
                info[k] = self._cumrewards[k]
            #
            # info['reward_carbon'] = self._cumrewards[0]
            # info['reward_species'] = self._cumrewards[1]

        else:
            self._cumrewards += reward
            info['reward'] = self._cumrewards

        self.history.append(
            [info["ExtantSpecies"], info["ExtantSpeciesValue"],
             info["ExtantSpeciesPD"], info["TotalCarbon"],
             info["TotPopulation"]
             ] + list(self.risk_label_counts(normalize=True)) + self.get_metrics() + [
                self._initialBudget - self.budget, self.n_species - info['non_protected_species'],
                np.sum(self.bioDivGrid.getCarbonValue_cell()) - self._init_total_carbon,
                np.sum(self.bioDivGrid.getCarbonValue_cell()[self.bioDivGrid.protection_matrix > 0])

            ]
        )
        if self._verbose and skip_env_step is False and self.calcReward is True:
            if self.rewardMode == "ext_risk_carbon":
                print("Reward:", reward, 'C:', reward_c, 'Sp:', reward_sp)
            elif self.rewardMode == "ext_risk_protect":
                print("Reward:", reward, reward_lab, reward_protect)
                print(self.risk_label_counts(normalize=True))
                print(self.current_risk)
                print(self.getExtinction_risk_labels())
                print(self.get_protected_fraction())
                print(np.sum(self.bioDivGrid.protection_matrix > 0), np.mean(self.bioDivGrid.protection_matrix))
            else:
                print("\nReward:", reward,
                      {key: round(value, 2) for key, value in self._cumrewards.items()},
                      self.risk_label_counts())
                # tmp = self.bioDivGrid.protectedRangePerSpecies() / (self.bioDivGrid.geoRangePerSpecies() + SMALL_NUMBER)
                # print(tmp[:10])
                # print(self.get_protected_fraction()[:10])
                # print(self.bioDivGrid.protectedRangePerSpecies()[:10])
                # try:
                #     print(self.species_risk_criteria.starting_rl_status[:10])
                # except: pass
        # richObs = state in RL
        # print("""state['grid_obj_most_recent']""", np.mean(richObs['grid_obj_most_recent'].h))
        return richObs, reward, done, info

    def _enrichObs(self):
        state = {"budget_left": self.budget}
        state["full_grid"] = self.bioDivGrid.h
        state["disturbance_matrix"] = self.bioDivGrid.disturbance_matrix
        state["selective_disturbance"] = self.bioDivGrid.selective_disturbance_matrix
        state["grid_obj_most_recent"] = self.grid_obj_most_recent
        state["grid_obj_previous"] = self.grid_obj_previous
        state["resolution"] = self.resolution
        if self._protection_cost is not None:
            # print("\nstate[protection_matrix]", np.sum(np.maximum(self.bioDivGrid.protection_matrix,
            #                                         self._protection_cost > self.budget)),
            #       np.sum(self.bioDivGrid.protection_matrix))
            state["protection_matrix"] = np.maximum(self.bioDivGrid.protection_matrix,
                                                    self._protection_cost > self.budget).astype(int) # cells w cost > budget are excluded
        else:
            state["protection_matrix"] = self.bioDivGrid.protection_matrix
        state["cost_quadrant"] = self._baseline_cost_quadrant + self.getProtectCostQuadrant()
        state["time_since_last_obs"] = self.timeSinceLastObserve
        if self.rewardMode == 3:  # (use PD loss) sp value -> PD contribution
            state["sp_values"] = self.bioDivGrid.get_sp_pd_contribution()
        state["sp_values"] = self.list_species_values
        state["min_pop_requirement"] = None
        state["met_prot_target"] = None
        state["species_threat_label"] = self.getExtinction_risk_labels()
        state["species_future_threat_label"] = self.getFutureExtinction_risk_labels()
        state["n_threat_labels"] = self.species_risk_criteria.n_labels
        state["quandrant_grid_indx"] = self._quandrant_grid_indx

        return state

    def reset(self, initTimeSinceLastObserve=5, fullInfo=False):
        self._initEnv()
        if self._verbose:
            print("Observe...")
        if self.ignoreFirstObs:
            self.observe(initTimeSinceLastObserve)
        else:
            self.observe()
        # build output by stacking obs and time till last obs
        if self._verbose:
            print("_enrichObs...")
        richObs = self._enrichObs()
        self.protected_quadrants = []
        if self.rewardMode == "ext_risk_carbon":
            self._cumrewards = np.zeros(2)
        elif self.rewardMode == "pareto_mlt":
            self._cumrewards = {str(key): float(0) for key in self._reward_weights.keys()}
            self._step_rewards = {str(key): float(0) for key in self._reward_weights.keys()}
            # print("\n\nself._cumrewards", self._cumrewards, self._reward_weights.keys())
            # print("self._step_rewards", self._step_rewards)

        else:
            self._cumrewards = np.zeros(1)

        if not fullInfo:
            return richObs
        else:
            info = self._getInfo()
            return richObs, 0, False, info


    def get_full_info(self):
        self.observe()
        richObs = self._enrichObs()
        info = self._getInfo()
        reward = 0
        return richObs, reward, False, info

    def reset_init_values(self):
        info = self._getInfo()
        self.n_extant_init = self.n_extant
        self.value_extant_sp_init = self.value_extant_sp
        self.pd_extant_sp_init = self.pd_extant_sp
        self._init_total_carbon = np.sum(self.bioDivGrid.getCarbonValue_cell())
        self._init_total_population = np.sum(self.bioDivGrid.h)
        self.currentIteration = 0
        self.bioDivGrid._counter = 0
        self.species_risk_criteria._init_range_size = self.bioDivGrid.geoRangePerSpecies() + 0
        self._init_sp_ext_risk = self.getExtinction_risk_labels()
        self.current_protected_range_fraction = np.sum(self.get_protected_fraction())
        # self.history = [[info["ExtantSpecies"], info["ExtantSpeciesValue"],
        #      info["ExtantSpeciesPD"], info["TotalCarbon"],
        #      info["TotPopulation"]] + list(self.risk_label_counts(normalize=True)) + self.get_metrics()]


        self.history = []

        if self.rewardMode == "ext_risk_carbon":
            self._cumrewards = np.zeros(2)
        elif self.rewardMode == "pareto_mlt":
            self._cumrewards = {key: 0 for key in self._reward_weights.keys()}
            self._step_rewards = {key: 0 for key in self._reward_weights.keys()}
        else:
            self._cumrewards = np.zeros(1)


    def reset_risk_labels(self):
        self.species_risk_criteria = self.ext_risk_class(self.bioDivGrid)

    def set_species_risk_criteria(self, ext_risk_obj):
        self.species_risk_criteria = ext_risk_obj
        self._init_sp_ext_risk = self.getExtinction_risk_labels()
        self.ext_risk_class = ext_risk_obj.__class__
        if DEBUG:
            print("BioDivEnv: self.ext_risk_class", self.ext_risk_class.__class__)


    def _getInfo(self):
        info = {
            "budget_not_done": self.budget > 0.0,
            "can_protect": self._canProtect(),
            "NumberOfProtectedCells": np.sum(self.bioDivGrid.protection_matrix > 0.0),
            "budget_left": self.budget,
            "time_last_protect": self.timeOfLastProtect,
            "CostOfProtection": self.cost_protected_quadrants,
            "ExtantSpecies": self.n_extant / self.n_extant_init,
            "ExtantSpeciesValue": self.value_extant_sp / self.value_extant_sp_init,
            "ExtantSpeciesPD": self.pd_extant_sp / self.pd_extant_sp_init,
            "TotalCarbon": np.sum(self.bioDivGrid.getCarbonValue_cell()) / self._init_total_carbon,
            "TotPopulation": np.sum(self.bioDivGrid.h) / self._init_total_population,
            "protection_matrix": self.bioDivGrid.protection_matrix
        }
        tmp = self.risk_label_counts(normalize=True)
        c_tmp = 0
        for n_lab in self.species_risk_criteria.available_labels_text:
            info[n_lab] = tmp[c_tmp]
            c_tmp += 1

        protected_range_fraction = self.bioDivGrid.protectedRangePerSpecies() / (self.bioDivGrid.geoRangePerSpecies() + SMALL_NUMBER)
        # average fraction of protected range per threat class (initial classification)
        c_tmp = 0
        for n_lab in self.species_risk_criteria.available_labels_text:
            if np.any(self._init_sp_ext_risk == c_tmp):
                protected_fr_risk_class = np.mean(protected_range_fraction[self._init_sp_ext_risk == c_tmp])
            else:
                protected_fr_risk_class = 0

            info[n_lab + "_pr"] = protected_fr_risk_class
            c_tmp += 1

        info['non_protected_species'] = len(protected_range_fraction[protected_range_fraction == 0])



        return info

    def render(self, mode="human_print", close=False):
        if mode == "human_print" and self._verbose:
            print(
                f"Iteration: {self.currentIteration}; Budget: {self.budget};"
                f" NumSpecies: {self.bioDivGrid.numberOfSpecies()}; NumIndividuals: {np.sum(self.bioDivGrid.h)}"
            )
        elif mode == "dict_csv_ready":
            return {
                "iteration": self.currentIteration,
                "budget": self.budget,
                "num_species": self.bioDivGrid.numberOfSpecies(),
                "num_individuals": np.sum(self.bioDivGrid.individualsPerSpecies()),
                "mean_disturbance": np.mean(self.bioDivGrid.disturbance_matrix),
                "mean_selective_disturbance": np.mean(
                    self.bioDivGrid.selective_disturbance_matrix
                ),
                "num_protected_cells": np.sum(self.bioDivGrid.protection_matrix > 0),
                "time_since_last_observe": self.timeSinceLastObserve,
                "last_action_type": self.lastActionType,
            }

        else:
            raise (NotImplementedError(f"mode {mode} not implemented!"))

    def _cellCoordinateFromIndex(self, actionValue):
        length = self.bioDivGrid.length
        col = actionValue % length
        row = int(actionValue / length)
        return row, col

    def reset_RunMode(self, mode):
        self.runMode = mode

    def set_species_cell_K(self):
        self.bioDivGrid.species_cell_specific_capacity(self.bioDivGrid.h)

    def heterogeneous_carrying_capacity(self,
                                        baseline_K=0.3, # minimum K (fraction of initial)
                                        seed=0,
                                        species_cell_K=False,
                                        verbose=False
                                        ):
        if seed == 0:
            seed = np.random.randint(1111, 9999)
        distb_obj, _ = get_disturbance(2, seed=seed)
        x = np.zeros(self.bioDivGrid.disturbance_matrix.shape)
        for i in range(20):
            x = distb_obj.updateDisturbance(x)
        if DEBUG:
            print("het K", np.mean(x), distb_obj._rr, distb_obj._counter)
        normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
        K_max = normalized * (1 - baseline_K) + baseline_K
        self.bioDivGrid.reset_K_max(K_max * self.bioDivGrid._K_max)
        self.bioDivGrid._immediate_capacity = 1
        tmp = copy.deepcopy(self.runMode)
        self.runMode = 0
        # _ = self.step()
        self.evolve(1, verbose=verbose)
        self.runMode = tmp
        self.bioDivGrid._immediate_capacity = 0
        self.reset_init_values()
        self.reset_risk_labels()
        if species_cell_K:
            self.set_species_cell_K()
        self.observe()
        self.observe() # observe x2 to update both grid_obj_most_recent and grid_obj_previous

    def evolve(self, steps=1, verbose=True):
        tmp = copy.deepcopy(self.runMode)
        tmp2 = copy.deepcopy(self._verbose)
        self.runMode = 0
        if not verbose:
            self._verbose = 0
        __ = [self.step() for _ in range(steps)]
        self.runMode = tmp
        self._verbose = tmp2

    def fast_forward(self,
                     steps=1,
                     disturbance_effect_multiplier=100,
                     verbose=True,
                     skip_dispersal=False,
                     additional_steps=0):
        tmp1 = copy.deepcopy(self.runMode)
        tmp2 = copy.deepcopy(self._verbose)
        tmp3 = copy.deepcopy(self.bioDivGrid.disturbance_effect_multiplier)
        self.runMode = 0
        if not verbose:
            self._verbose = 0
        self.bioDivGrid.disturbance_effect_multiplier = disturbance_effect_multiplier
        print("running fast_forward:", steps, additional_steps)
        __ = [self.step(skip_dispersal=skip_dispersal) for _ in range(steps)]
        print("done")
        self._verbose = tmp2
        self.bioDivGrid.disturbance_effect_multiplier = tmp3
        if additional_steps:
            print("running additional_steps: ", additional_steps)
            for _ in range(additional_steps):
                __ = self.step(skip_dispersal=False)
                # self.species_risk_criteria.update_pop_sizes(self.bioDivGrid)
        self.runMode = tmp1 # reset runmode to PROTECT/ORACLE

    def get_net_carbon(self):
        # normalized by a constant: _init_total_carbon, per cell
        tmp = np.sum(self.bioDivGrid.getCarbonValue_cell()) - self.current_carbon
        return tmp / self._init_total_carbon * self.bioDivGrid.protection_matrix.size

    def set_total_natural_carbon(self, c):
        self._total_natural_carbon = c

    def set_reward_weights(self, rw):
        if self._verbose:
            print("Setting reward_weights", rw)
        self._reward_weights = rw
        self._cumrewards = {str(key): float(0) for key in self._reward_weights.keys()}
        self._step_rewards = {str(key): float(0) for key in self._reward_weights.keys()}

    def set_grid_obj_h_previous(self, grid_obj_h):
        self.grid_obj_previous._h = grid_obj_h

    def set_calc_reward(self, r):
        self.calcReward = r

    def get_metrics(self):
        msa = calc_MSA_from_grid(self.grid_obj_previous.h, self.grid_obj_most_recent.h)
        return [msa]

    def set_max_protection_level(self, m):
        self._max_protection_level = m

    def set_costs(self, cost_layer, cost_coef=1, baseline_cost_quadrant=0):
        self._protection_cost = cost_layer
        self._cost_coeff = cost_coef
        self._baseline_cost_quadrant = baseline_cost_quadrant * np.ones(cost_layer.size)

    def set_budget(self, b):
        self.budget = b
        self._initialBudget = b