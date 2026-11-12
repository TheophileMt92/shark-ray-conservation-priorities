import sys

import numpy as np

from numba import jit
import random
import scipy.stats
import sparse
from collections.abc import Iterable


small_number = 1e-10
DEBUG = False

@jit(nopython=True)
def dispersalDistances(length, lambda_0):
    print("calculating distances...")
    dumping_dist = np.zeros((length, length, length, length))
    for i in range(0, length):
        for j in range(0, length):
            for n in range(0, length):
                for m in range(0, length):
                    exp_rate = 1.0 / lambda_0
                    # relative dispersal probability: always 1 at distance = 0
                    # the actual number of offspring is modulated by growth_rate
                    dumping_dist[i, j, n, m] = np.exp(
                        -exp_rate * np.sqrt((i - n) ** 2 + (j - m) ** 2)
                    )
    return dumping_dist


@jit(nopython=True)
def dispersalDistancesThreshold(length: int,
                                lambda_0: float,
                                threshold=3):
    print("calculating distances with threshold...")
    dumping_dist = np.zeros((length, length, length, length))
    for i in range(0, length):
        for j in range(0, length):
            for n in range(max([0, i-threshold]), min([length, i+threshold])):
                for m in range(max([0, j-threshold]), min([length, j+threshold])):
                    exp_rate = 1.0 / lambda_0
                    # relative dispersal probability: always 1 at distance = 0
                    # the actual number of offspring is modulated by growth_rate
                    dumping_dist[i, j, n, m] = np.exp(
                        -exp_rate * np.sqrt((i - n) ** 2 + (j - m) ** 2)
                    )
    return dumping_dist


@jit(nopython=True)
def dispersalDistancesThresholdCoord(length: int,
                                     lambda_0: float,
                                     lat: np.ndarray,
                                     lon: np.ndarray,
                                     threshold=3
                                     ):
    print("calculating distances with threshold...")
    exp_rate = 1.0 / lambda_0
    dumping_dist = np.zeros((length, length, length, length), dtype=np.float32)
    for i in range(0, length):
        for j in range(0, length):
            # cn.print_update("%s / %s, %s" % (i, length, j))
            for n in range(0, length):
                for m in range(0, length):
                    # print(abs(lat[i,j] - lat[n,m]), abs(lon[i,j] - lon[n,m]) )
                    if abs(lat[i,j] - lat[n,m]) <= threshold and abs(lon[i,j] - lon[n,m]) <= threshold:
                        # relative dispersal probability: always 1 at distance = 0
                        # the actual number of offspring is modulated by growth_rate
                        # print(i, j, n, m)
                        dumping_dist[i, j, n, m] = np.exp(
                            -exp_rate * np.sqrt((lat[i,j] - lat[n,m]) ** 2 + (lon[i,j] - lon[n,m]) ** 2)
                        )
    return dumping_dist


@jit(nopython=True)
def dispersalDistancesThresholdCoord2D(length: int,
                                       lambda_0: float,
                                       lat_i: float,
                                       lon_i: float,
                                       lat: np.ndarray,
                                       lon: np.ndarray,
                                       # dumping_dist: np.ndarray,
                                       threshold=3,

                                       ):
    exp_rate = 1.0 / lambda_0
    dumping_dist = np.zeros((length, length), dtype=np.float32)
    for n in range(0, length):
        for m in range(0, length):
            # print(abs(lat[i,j] - lat[n,m]), abs(lon[i,j] - lon[n,m]) )
            if abs(lat_i - lat[n, m]) <= threshold and abs(lon_i - lon[n, m]) <= threshold:
                # relative dispersal probability: always 1 at distance = 0
                # the actual number of offspring is modulated by growth_rate
                # print(i, j, n, m)
                dumping_dist[n, m] = np.exp(
                    -exp_rate * np.sqrt((lat_i - lat[n, m]) ** 2 + (lon_i - lon[n, m]) ** 2)
                )
    return dumping_dist


def dispersalDistancesThresholdCoord4D(length: int,
                                     lambda_0: float,
                                     lat: np.ndarray,
                                     lon: np.ndarray,
                                     threshold=3
                                     ):
    print("calculating distances with threshold...")
    exp_rate = 1.0 / lambda_0
    dumping_dist = np.zeros((length, length, length, length), dtype=np.float32)
    for i in range(0, length):
        for j in range(0, length):
            cn.print_update("%s / %s, %s" % (i, length, j))


            x = dispersalDistancesThresholdCoord2D(length, lambda_0, lat[i, j], lon[i, j], lat, lon,
                                                   np.zeros((length, length), dtype=np.float32), threshold)
            dumping_dist[i, j] = x + 0
            del x #, dispersalDistancesThresholdCoord2D
    return dumping_dist



def dispersalDistancesThresholdCoord_vectorized(length: int,
                                                 lambda_0: float,
                                                 lat: np.ndarray, # Assuming lat has shape (length, length)
                                                 lon: np.ndarray, # Assuming lon has shape (length, length)
                                                 threshold: int = 3
                                                 ) -> np.ndarray: # Specify return type hint
    print("calculating distances with threshold (vectorized, with lat/lon)...")

    exp_rate = 1.0 / lambda_0

    # Reshape lat and lon arrays for broadcasting
    # lat_ij: (length, length, 1, 1) to represent lat[i,j] for all i,j
    # lon_ij: (length, length, 1, 1) to represent lon[i,j] for all i,j
    lat_ij = lat.reshape(length, length, 1, 1)
    lon_ij = lon.reshape(length, length, 1, 1)

    # lat_nm: (1, 1, length, length) to represent lat[n,m] for all n,m
    # lon_nm: (1, 1, length, length) to represent lon[n,m] for all n,m
    lat_nm = lat.reshape(1, 1, length, length)
    lon_nm = lon.reshape(1, 1, length, length)

    # Calculate differences in latitude and longitude using broadcasting
    # These will both have shape (length, length, length, length)
    delta_lat = lat_ij - lat_nm
    delta_lon = lon_ij - lon_nm

    # Calculate squared Euclidean distance
    squared_distance = delta_lat**2 + delta_lon**2
    distance = np.sqrt(squared_distance)

    # Calculate the exponential term for all combinations
    # This will also have shape (length, length, length, length)
    dumping_dist_full = np.exp(-exp_rate * distance)

    # --- Apply the threshold condition ---
    # Create boolean masks based on absolute differences exceeding threshold
    # These masks will also have shape (length, length, length, length)
    lat_within_threshold = np.abs(delta_lat) <= threshold
    lon_within_threshold = np.abs(delta_lon) <= threshold

    # Combine the masks using logical AND.
    # An element is True only if *both* lat and lon differences are within threshold.
    combined_mask = lat_within_threshold & lon_within_threshold

    # Initialize the result array with zeros (outside the loop where it's not needed)
    # This array will hold the final dumping_dist values, with non-threshold values as 0
    result_array = np.zeros((length, length, length, length), dtype=np.float32)

    # Use the combined_mask to selectively assign values from dumping_dist_full
    # Only elements where combined_mask is True will get the calculated value;
    # others remain 0 as initialized.
    result_array[combined_mask] = dumping_dist_full[combined_mask]

    return result_array











def add_random_diffusion_mortality(
    length, sig=5, peak_disturbance=0.3, min_death_prob=0.01
):
    indx = np.meshgrid(np.arange(length), np.arange(length))
    locsxy = np.random.uniform(0, length, (2, 1))
    min_sig2 = 1
    sig_tmp = np.random.uniform(min_sig2, sig, 2)
    # print("\n\nlocsxy:", locsxy, "\n")
    disturbance_matrix_tmp = scipy.stats.norm.pdf(
        indx[0], loc=locsxy[0, 0], scale=sig_tmp[0]
    ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, 0], scale=sig_tmp[1])

    disturbance_matrix_tmp = (
        disturbance_matrix_tmp / np.max(disturbance_matrix_tmp)
    ) * peak_disturbance + min_death_prob
    disturbance_matrix_tmp[disturbance_matrix_tmp > 0.99] = 1 - small_number
    return disturbance_matrix_tmp


def add_random_error(probs, sig=0.1):
    rates = -np.log(1 - probs)
    log_rates = np.log(rates)
    tmp_log_rates = np.random.normal(0, sig * log_rates, probs.shape)
    rnd_log_rates = log_rates + tmp_log_rates
    probs = 1 - np.exp(-np.exp(rnd_log_rates))
    probs = np.maximum(probs, np.zeros(rates.shape) + small_number)
    probs = np.minimum(probs, np.ones(rates.shape) - small_number)
    return probs


def add_random_error_per_species(probs, sig=0.1):
    rates = -np.log(1 - probs)
    log_mean_rates = np.abs(np.log(np.mean(rates)))
    tmp_rates = np.exp(np.random.normal(0, sig * log_mean_rates, probs.shape[0]))
    rnd_rates = np.einsum("sij,s-> sij", rates, tmp_rates)
    probs = 1 - np.exp(-rnd_rates)
    probs = np.maximum(probs, np.zeros(rates.shape) + small_number)
    probs = np.minimum(probs, np.ones(rates.shape) - small_number)
    return probs


def get_alpha_K(probs, N_K_ratio):
    rates = -np.log(1 - probs)
    new_rates = np.maximum((rates * (N_K_ratio - 1)), np.zeros(rates.shape))
    probs = 1 - np.exp(-new_rates)
    # print(np.max(probs),np.max(N_K_ratio),np.min(probs),np.min(N_K_ratio))
    return probs

def get_alpha_K_species_cell(probs, N_K_ratio):
    rates = -np.log(1 - probs)
    N_K_ratio_n = np.minimum((N_K_ratio - 1), np.ones(rates.shape))
    new_rates = np.maximum((rates * N_K_ratio_n), np.zeros(rates.shape))
    probs = 1 - np.exp(-new_rates)
    # print(np.max(probs),np.max(N_K_ratio),np.min(probs),np.min(N_K_ratio))
    return probs


def get_alpha_K_(probs, N_K_ratio):
    prob_death = np.zeros(probs.shape)
    # print( N_K_ratio.shape,prob_death.shape, N_K_ratio )
    prob_death[:, N_K_ratio > 1] = probs[:, N_K_ratio > 1]
    return prob_death

class empty_tree:
    def length(self):
        return 0

def extract_tree_with_taxa_labels(tree, labels):
    try:
        subtree = tree.extract_tree_with_taxa_labels(labels=labels)
    except:
        subtree = empty_tree()
    return subtree


class SimGrid(object):
    def __init__(
            self,
            length: int,
            num_species: int,
            alpha: float,
            K_max: float,
            lambda_0: object,
            disturbanceInitializer: object,
            disturbance_sensitivity: object,
            selectivedisturbanceInitializer: object = 0,
            selective_sensitivity: object = [],
            immediate_capacity: object = False,
            truncateToInt: object = False,
            species_threshold: object = 1,
            rnd_alpha: object = 0,
            K_disturbance_coeff: object = 1,
            actions: object = [],
            dispersal_before_death: object = 0,
            rnd_alpha_species: object = 0,
            climateModel: object = 0,
            growth_rate: object = np.ones(1),
            phyloGenerator: object = 0,
            climate_sensitivity: object = [],
            climate_as_disturbance=1,
            disturbance_dep_dispersal=1,
            species_cell_specific_capacity=None,
            habitat_suitability=None,
            future_habitat_suitability=None,
            delta_suitability_per_step=None,
            species_threshold_per_cell=1,
            precomputed_dispersal_probs=None,
            K_species=None, # max number of individuals of a species per cell
            rm_lingering_pops=False
    ):
        self._length = length
        self._n_species = num_species
        self._species_id = np.arange(num_species)
        self._alpha = alpha  # fraction killed (1 number)
        self._K_max = K_max * np.ones((self._length, self._length))  # initial (max) carrying capacity
        self._lambda_0 = (
            lambda_0  # relative dispersal probability: always 1 at distance = 0
        )
        try:
            if len(growth_rate) < num_species:
                self._growth_rate = np.ones(num_species) * growth_rate
            else:
                self._growth_rate = growth_rate  # potential number of offspring per individual per year at distance = 0
        except:
            self._growth_rate = np.ones(num_species) * growth_rate
        self._disturbanceInitializer = disturbanceInitializer
        self._disturbance_matrix = np.zeros((self._length, self._length))
        self._K_cells = (1 - self._disturbance_matrix) * self._K_max
        self._K_disturbance_coeff = (
            K_disturbance_coeff  # if set to 0.5, K is 0.5*(1-disturbance)
        )
        self._counter = 0
        self._species_threshold = species_threshold
        self._species_threshold_per_cell = species_threshold_per_cell
        self._dispersal_before_death = dispersal_before_death  # set to 1/0 to get dispersing pool before/after death

        self._disturbance_sensitivity = (
            disturbance_sensitivity  # vector of sensitivity per species
        )
        self._alpha_histogram = self.alphaHistogram(
            self._disturbance_sensitivity, self._disturbance_matrix
        )
        self._rnd_alpha = rnd_alpha
        self._rnd_alpha_species = rnd_alpha_species
        self._immediate_capacity = immediate_capacity
        self._truncateToInt = truncateToInt

        if len(actions) == 0:
            self._selective_disturbance_matrix = np.zeros((self._length, self._length))
            self._protection_matrix = np.zeros((self._length, self._length))
        else:
            self._selective_disturbance_matrix = actions[0]
            self._protection_matrix = actions[1]
        self._selectivedisturbanceInitializer = selectivedisturbanceInitializer

        self._selective_sensitivity = (
            selective_sensitivity  # vector of selective sensitivity per species
        )
        self._selective_alpha_histogram = self.alphaHistogram(
            self._selective_sensitivity, self._selective_disturbance_matrix
        )
        # self.updateSelectiveAlphaHistogram() #TODO check if you can use this instead
        # TODO do we need to do this also? self._h = self._h * (1 - self._selective_alpha_histogram)
        # self._alpha_by_cell = np.ones((length,length))
        self._climate_sensitivity = climate_sensitivity
        self._climate_as_disturbance = climate_as_disturbance
        self._disturbance_dep_dispersal = disturbance_dep_dispersal
        self._disturbance_matrix_diff = 0

        if climateModel == 0:
            self._climateModel = 0
            self._climate_layer = np.zeros((self._length, self._length))
        else:
            self._climateModel = climateModel
            self._climateModel.reset_counter()
            if self._climate_as_disturbance:
                self._climate_layer = self._climateModel.updateClimate(
                    np.zeros((self._length, self._length))
                )
            else:
                self._climate_layer = self._climateModel.updateClimate(
                    np.ones((self._length, self._length))
                )

        # TODO: remove dependency on phylo data?
        if phyloGenerator == 0:
            from ..biodivinit.PhyloGenerator import ReadRandomPhylo as phyloGenerator

            try:
                self._phyloGenerator = phyloGenerator(
                    phylofolder="data_dependencies/phylo/"
                )
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = self._phyloGenerator.getPhylo()
            except:
                from ..biodivinit.PhyloGenerator import SimRandomPhylo as phyloGenerator

                self._phyloGenerator = phyloGenerator(n_species=self._n_species)
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = self._phyloGenerator.getPhylo()
        else:
            try:
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = phyloGenerator.getPhylo()
            except:
                from ..biodivinit.PhyloGenerator import SimRandomPhylo as phyloGenerator

                self._phyloGenerator = phyloGenerator(n_species=self._n_species)
                (
                    self._phylo_tree,
                    self._all_tip_labels,
                    self._phylo_ed,
                    self._phylo_file_name,
                ) = self._phyloGenerator.getPhylo()

        self._species_cell_specific_capacity = species_cell_specific_capacity
        self._K_species_cells = species_cell_specific_capacity
        self.species_carbon_value = np.ones(num_species)
        self.disturbance_effect_multiplier = 1 # >1 to fast-forward effect of disturbance
        self.additional_carbon_matrix = np.zeros((self._length, self._length))
        self._habitat_suitability = habitat_suitability
        self._future_habitat_suitability = future_habitat_suitability
        self._delta_suitability_per_step = None
        self._dumping_dist = precomputed_dispersal_probs
        self._K_species = K_species
        if K_species is not None and habitat_suitability is not None:
            self._K_species3D = K_species[:, np.newaxis, np.newaxis] * habitat_suitability
        else:
            self._K_species3D = None
        self._reference_grid_pu = None
        self._rm_lingering_pops = rm_lingering_pops
        self._n_pus = None


    def get_sp_pd_contribution(self):
        totalpd = self._phylo_tree.length()
        phylo_ed = np.zeros(self._n_species)
        # print(len(self._all_tip_labels))
        c = 0
        for i in self._all_tip_labels:
            subtree = extract_tree_with_taxa_labels(self._phylo_tree,
                                                    labels=self._all_tip_labels[self._all_tip_labels != i])
            # subtree = self._phylo_tree.extract_tree_with_taxa_labels(
            #     labels=self._all_tip_labels[self._all_tip_labels != i]
            # )
            phylo_ed[c] = totalpd - subtree.length()
            c += 1
        self._phylo_ed = phylo_ed / np.sum(phylo_ed) * self._n_species

    def alphaHistogram(self, disturbanceSensitivity, disturbanceMatrix):
        # TODO: implement 3D disturbance
        # print("disturbanceSensitivity.shape", disturbanceSensitivity)
        # if len(disturbanceSensitivity.shape) == 2:
        #     "when alphaHistogram==0: nobody dies, when==1: all die"
        #     return np.einsum("sd,dij->sij", disturbanceSensitivity, disturbanceMatrix)
        # if len(disturbanceSensitivity.shape) == 1:
        "when alphaHistogram==0: nobody dies, when==1: all die"
        return np.einsum("s,ij->sij", disturbanceSensitivity, disturbanceMatrix)

    def setProtectionMatrix(self, protection_matrix):
        self._protection_matrix = protection_matrix

    def setSelectiveDisturbanceMatrix(self, selective_disturbance_matrix):
        self._selective_disturbance_matrix = selective_disturbance_matrix

    def setDisturbanceMatrix(self, disturbance_matrix):
        self._disturbance_matrix = disturbance_matrix

    def updateAlphaHistogram(self):
        new_dist = self._disturbanceInitializer.updateDisturbance(
            self._disturbance_matrix
        )
        self._disturbance_matrix_diff = np.mean(new_dist - self._disturbance_matrix)
        self._disturbance_matrix = new_dist
        self._alpha_histogram = self.alphaHistogram(
            self._disturbance_sensitivity,
            self._disturbance_matrix * (1 - self._protection_matrix),
        )

        if self._rnd_alpha > 0:
            self._alpha_histogram = add_random_error(
                self._alpha_histogram, sig=self._rnd_alpha
            )
        if self._rnd_alpha_species > 0:
            self._alpha_histogram = add_random_error_per_species(
                self._alpha_histogram, sig=self._rnd_alpha_species
            )

    def get_species_mid_coordinate(self):
        med_lat, med_lon = [], []
        for sp_i in range(self._n_species):
            tmp = np.sum(self._h[sp_i, :, :], axis=1)
            lat_range = np.where(tmp > 0)
            tmp = np.sum(self._h[sp_i, :, :], axis=0)
            lon_range = np.where(tmp > 0)
            med_lat.append(np.median(lat_range))
            med_lon.append(np.median(lon_range))
            # print(i, median_latitude, median_longitude)
        return np.array([med_lat, med_lon])
    
    def set_species_carbon_value(self, val):
        self.species_carbon_value = val

    def getCarbonValue_cell(self):
        natural_carbon = np.einsum('sxy,s -> xy', self.h, self.species_carbon_value)
        return natural_carbon + self.additional_carbon_matrix

    def set_habitat_suitability(self, suit):
        self._habitat_suitability = suit

    def set_future_habitat_suitability(self, suit):
        self._future_habitat_suitability = suit

    def set_delta_suitability_per_step(self, delta_suitability):
        self._delta_suitability_per_step = delta_suitability

    def getClimateTolerance(self):
        temp = self._h + 0
        temp[temp > 1] = 1
        temp[temp < 1] = 0
        max_T = np.zeros(self._n_species) + np.max(self._climate_layer)
        min_T = np.zeros(self._n_species) + np.min(self._climate_layer)
        for sp_i in range(self._n_species):
            if np.sum(temp[sp_i, :, :]) > 0:
                max_T[sp_i] = np.max(self._climate_layer[temp[sp_i, :, :] == 1])
                min_T[sp_i] = np.min(self._climate_layer[temp[sp_i, :, :] == 1])

        climate_tolerance_range = np.array(
            [min_T, max_T]
        ).T  # 2D array: species x 2 (min/max)
        # for species reaching the boundaries make up tolerance ranges wider than the full grid
        rnd_ranges = np.random.uniform(
            0,
            np.max(self._climate_layer) - np.min(self._climate_layer),
            climate_tolerance_range.shape,
        )
        climate_tolerance_range[
            climate_tolerance_range[:, 1] == np.max(self._climate_layer), 1
        ] += rnd_ranges[climate_tolerance_range[:, 1] == np.max(self._climate_layer), 1]
        climate_tolerance_range[
            climate_tolerance_range[:, 0] == np.min(self._climate_layer), 0
        ] -= rnd_ranges[climate_tolerance_range[:, 0] == np.min(self._climate_layer), 0]
        # mid-point, half-range
        climate_tolerance = np.array(
            [
                np.mean(climate_tolerance_range, axis=1),
                np.diff(climate_tolerance_range, axis=1)[:, 0] / 2.0,
            ]
        ).T
        n = np.ones(self._climate_layer.shape)
        # 3D: species x long x lat (values repeated for each species across all cells)
        climate_opt_sp_3D = np.einsum("s,ij -> sij", climate_tolerance[:, 0], n)
        climate_range_sp_3D = np.einsum("s,ij -> sij", climate_tolerance[:, 1], n)
        return climate_opt_sp_3D, climate_range_sp_3D

    def updateSelectiveAlphaHistogram(self):
        if self._selectivedisturbanceInitializer != 0:
            a = self._selective_disturbance_matrix + 0
            self._selective_disturbance_matrix = (
                self._selectivedisturbanceInitializer.updateDisturbance(
                    self._selective_disturbance_matrix
                )
            )
            # only delta change considered
            self._selective_alpha_histogram = self.alphaHistogram(
                self._selective_sensitivity,
                (self._selective_disturbance_matrix - a) * (1 - self._protection_matrix),
            )


        else:  # in this case selective_disturbance = disturbance
            self._selective_disturbance_matrix = self._disturbance_matrix

            self._selective_alpha_histogram = self.alphaHistogram(
                self._selective_sensitivity,
                self._selective_disturbance_matrix * (1 - self._protection_matrix),
            )

    # def getAlphaByCell(self):
    # 	self._alpha_by_cell = self._alpha_by_cell * self._alpha
    # 	self._alpha_by_cell = add_random_error(self._alpha_by_cell,sig=self._rnd_alpha)

    def updateKcells(self):
        # carrying capacity should change with disturbance
        self._K_cells = (
            (1 - (self._disturbance_matrix * (1 - self._protection_matrix)))
            * self._K_disturbance_coeff
        ) * self._K_max

    def updateKspecies_cell(self):
        if self._species_cell_specific_capacity is not None:
            self._K_species_cells = (
                (1 - (self._disturbance_matrix * (1 - self._protection_matrix)))
                * self._K_disturbance_coeff
            ) * self._species_cell_specific_capacity


    def reset_K_max(self, new_K_max):
        self._K_max = new_K_max
        self.updateKcells()

    def totalCapacity(self):
        return np.einsum("ij->", self._K_cells)

    def initGrid(self, stateInitializer):
        print("\nself._dumping_dist", self._dumping_dist)

        # random histogram
        self._h = stateInitializer.getInitialState(
            self._K_max, self._n_species, self._length
        )
        # init dumping factors (unless already provided)
        if self._dumping_dist is None:
            self._dumping_dist = dispersalDistances(self._length, self._lambda_0)

        self.updateAlphaHistogram()
        self._climate_opt_sp_3D, self._climate_range_sp_3D = self.getClimateTolerance()
        if self._disturbance_dep_dispersal:
            sys.exit("Disturbance-dependent dispersal not implemented")
            # self._diag_list = getDiag.get_diagonals_from_pickle("../scripts/diagonals50.pkl")

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

    def protectedRangePerSpecies(self):
        return np.einsum("sij, ij -> s", self._h >= self._species_threshold_per_cell,
                         self._protection_matrix)

    def protectedFutureRangePerSpecies(self):
        return np.einsum("sij, ij -> s", self.future_h >= 0.5,
                         self._protection_matrix)

    def individualsPerCell(self):
        return np.einsum("sij->ij", self._h)

    def speciesPerCell(self):
        presence_absence = self._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        return np.einsum("sij->ij", presence_absence)

    def pdPerCell(self):  # calculate phylogenetic diversity (superslow)
        pd_grid = np.zeros((self._length, self._length))
        for i in range(0, self._length):
            for j in range(0, self._length):
                tmp_sp = self._h[:, i, j]
                if len(tmp_sp[tmp_sp > 1]) >= 2:
                    labels = self._all_tip_labels[tmp_sp > 1]
                    tree_cell = extract_tree_with_taxa_labels(
                        self._phylo_tree,
                        labels=labels
                    )
                    pd_grid[i, j] = tree_cell.length()
                else:
                    pd_grid[i, j] = 0
        return pd_grid

    def edPerCell(self):  # calculate evolutionary distinctiveness
        presence_absence = self._h + 0
        presence_absence[
            presence_absence < 1
        ] = 0  # species_threshold is only used for total pop size
        presence_absence[presence_absence > 1] = 1  # not within each cell
        return np.einsum("sij,s->ij", presence_absence, self._phylo_ed)

    def edPerSpecies(self):
        return self._phylo_ed

    def numberOfSpecies(self):
        return np.sum(np.einsum("sij->s", self._h) > self._species_threshold)

    def extantSpeciesID(self):
        return self._species_id[np.einsum("sij->s", self._h) > self._species_threshold]

    def extinctSpeciesID(self):
        return self._species_id[np.einsum("sij->s", self._h) < self._species_threshold]

    def totalEDextantSpecies(self):
        return np.sum(
            self._phylo_ed[np.einsum("sij->s", self._h) > self._species_threshold]
        )

    def totalPDextantSpecies(self):
        if self._phylo_tree == 0:
            return 1
        labels = self._all_tip_labels[self.extantSpeciesID()]
        tree_extant = extract_tree_with_taxa_labels(self._phylo_tree, labels=labels)
        pd_extant = tree_extant.length()
        pd_extant += small_number
        return pd_extant

    def numberOfIndividuals(self):
        return np.einsum("sij->", self._h)

    def geoRangePerSpecies(self, sp_threshold=None):  # number of occupied cells
        if sp_threshold is None:
            return np.einsum('sij->s', (self.h > self._species_threshold_per_cell).astype(float))
        else:
            return np.einsum('sij->s', (self.h > sp_threshold).astype(float))

    def geoFutureRangePerSpecies(self, sp_threshold=0.5):  # number of occupied cells
        if sp_threshold is None:
            return np.einsum('sij->s', (self.future_h > self._species_threshold_per_cell).astype(float))
        else:
            return np.einsum('sij->s', (self.future_h > sp_threshold).astype(float))


    def histogram(self):
        return self._h

    def update_dumping_dist(self, fast=False):
        # print(self._disturbance_matrix_diff, np.mean(self._dumping_dist))
        multiplier = 2
        self._dumping_dist = np.exp(
            np.log(self._dumping_dist + small_number)
            - (self._disturbance_matrix_diff * self._lambda_0) * multiplier
        )
        # print(self._disturbance_matrix_diff, np.mean(self._dumping_dist))

    def update_additional_carbon_matrix(self,
                                        plnt_matrix, # zeros nad ones
                                        steps=1,
                                        max_carb_per_cell=None,
                                        growth_rate=0.2,
                                        clear_threshold=0.95  # fraction at which plantation is cleared
                                        ):
        """
        defines carbon in plantation areas
        """
        if max_carb_per_cell is None:
            max_carb_per_cell = np.max(self.getCarbonValue_cell())
        # plnt_matrix = np.zeros((env_length, env_length))
        # plnt_matrix += 1  # set all matrix to plantation for now
        # carb_matrix = self.update_additional_carbon_matrix

        for i in range(steps):
            # max_carb_per_cell * plnt_matrix -> where there is no plantation 0 additional carbon
            self.additional_carbon_matrix = self.additional_carbon_matrix + growth_rate * \
                                                   (max_carb_per_cell * plnt_matrix - self.additional_carbon_matrix)
            self.additional_carbon_matrix[self.additional_carbon_matrix >= clear_threshold * max_carb_per_cell] = 0
            # print(i, self.update_additional_carbon_matrix[0, 0])

    def update_K_species3D(self):
        self._K_species3D = self._K_species[:, np.newaxis, np.newaxis] * self._habitat_suitability


    def update_habitat_suitability(self):
        # print("\nupdate_habitat_suitability",
        #       np.mean((self._habitat_suitability - (self._habitat_suitability + self._delta_suitability_per_step))**2))
        suit_tmp = np.maximum(0, self._habitat_suitability + self._delta_suitability_per_step['delta'])
        suit_tmp[suit_tmp < self._delta_suitability_per_step['threshold']] = 0
        self.set_habitat_suitability(suit_tmp)
        self.update_K_species3D()

    def step(self, action=None, fast_dist=False, skip_dispersal=False, update_suitability=False):
        # if self._counter == 0:
        # 	print(self._disturbance_sensitivity[0:5])
        # evolve the grid one time step
        if DEBUG:
            print("getting NumCandidates")
        if self._dispersal_before_death == 1:
            NumCandidates = np.einsum("sij,ijnm->snm", self._h, self._dumping_dist)
            normCandidates = NumCandidates / np.einsum("sij->ij", NumCandidates)
        # update alpha hist (only 1st step for now)
        if DEBUG:
            print("running updateAlphaHistogram")
        self.updateAlphaHistogram()

        # update carrying capacity
        if DEBUG:
            print("update carrying capacity")

        self.updateKcells()
        self.updateKspecies_cell()
        # kill individuals based on new carrying capacity
        if self._immediate_capacity:
            if self._species_cell_specific_capacity is not None:
                self._h = self._K_species_cells + 0
                # kill individuals based on natural death rate + disturbance
                self._h = self._h * (1 - self._alpha_histogram)
            else:
                # self._h = self._h * (self._K_cells / self._K_max)
                self._h = (self._h / np.einsum("sij->ij", self._h)) * self._K_max
                # kill individuals based on natural death rate + disturbance
                self._h = self._h * (1 - self._alpha_histogram)
        elif self._K_species is not None:
            self._h = self._h * (1 - self._alpha_histogram)
            if self._K_species3D is not None:
                self._h = np.minimum(self._h, self._K_species3D)
        else:
            # if self._species_cell_specific_capacity is not None:
            #     final_alpha_histogram = get_alpha_K(
            #         self._alpha_histogram,
            #         self._h / self._K_species_cells
            #     )
            #     self._h = self._h * (1 - final_alpha_histogram)
            # else:
            for _ in range(self.disturbance_effect_multiplier):
                final_alpha_histogram = get_alpha_K(
                    self._alpha_histogram,
                    self.individualsPerCell() / (self._K_cells + small_number)
                )
                # final_alpha_histogram = self._alpha_histogram * (self.individualsPerCell()-self._K_cells)/self._K_cells

                self._h = self._h * (1 - final_alpha_histogram)

        if DEBUG:
            print("running updateSelectiveAlphaHistogram")
        if len(self._selective_sensitivity) > 0:
            # update selective alpha hist
            self.updateSelectiveAlphaHistogram()
            self._h = self._h * (1 - self._selective_alpha_histogram)

        # climate change effects
        if DEBUG:
            print("killing individuals")

        if self._climateModel != 0:
            # update climate layer
            self._climate_layer = self._climateModel.updateClimate(self._climate_layer)

            if self._climate_as_disturbance:
                """
                climate model as a regional disturbance C ~ (0,1):
                - species sensitivities as tolerance thresholds t ~ (0,1)
                - if C > t: death_rate = C/t - 1
                death_rate = np.max(0, C/t - 1)
                death_prob = 1 - np.exp(-death_rate)
                """
                death_rate = (
                    np.einsum(
                        "s,ij->sij", 1 / self._climate_sensitivity, self._climate_layer
                    )
                    - 1
                )
                death_rate[death_rate < 0] = 0
                death_prob = 1 - np.exp(-death_rate)

                # print("\n\n\n")
                # print(self._climate_layer)
                # print(death_prob.shape)
                # print(death_prob)
                # print(death_rate)

                self._h = self._h * (1 - death_prob)
            else:
                # Gradual climate change with N-S gradient
                # # distance from optimal climate, then if > range: change death rate
                abs_dist_from_opt = np.abs(
                    self._climate_opt_sp_3D - self._climate_layer
                )
                # print("self._climate_layer", self._climate_layer)
                # (1 over 2*t_range) times distance from the range boundary
                # convert to probability
                tempAlpha = 1 - np.exp(
                    -(1 / (2 * self._climate_range_sp_3D))
                    * (abs_dist_from_opt - self._climate_range_sp_3D)
                )
                # # set to 0 death rate within range
                tempAlpha[abs_dist_from_opt <= self._climate_range_sp_3D] = 0
                # print(abs_dist_from_opt[96,:,0]-self._climate_range_sp_3D[96,:,0])
                # print((1/(2*self._climate_range_sp_3D[96,:,0]))*(abs_dist_from_opt[96,:,0]-self._climate_range_sp_3D[96,:,0]))
                # print( tempAlpha[96,:,0], self._climate_range_sp_3D[96,:,0] )
                # quit()
                self._h = self._h * (1 - tempAlpha)

        # kill anyway by natural death and replace those
        if self._rnd_alpha_species == 0:
            red_hist = self._h * (1 - self._alpha)
        else:
            by_species = 0
            if by_species:
                # add by-species randomness
                tmp_alpha = np.ones(self._n_species) - self._alpha
                tmp_alpha = add_random_error(tmp_alpha, sig=self._rnd_alpha_species)
                # print(tmp_alpha[0:5])
                red_hist = np.einsum("sij,s -> sij", self._h, tmp_alpha)
            else:
                rr = np.random.uniform(0.95, 1)
                tmp_alpha = add_random_diffusion_mortality(
                    self._length,
                    sig=self._rnd_alpha_species,
                    peak_disturbance=rr,
                    min_death_prob=self._alpha,
                )
                # print(tmp_alpha, self._alpha, rr, np.max(tmp_alpha))
                red_hist = np.einsum("sij,ij -> sij", self._h, 1 - tmp_alpha)

        self._h = red_hist

        # how many can be replaced: max( new_carrying capacity - current population, 0)
        if self._species_cell_specific_capacity is None:
            available_space = np.maximum(
                self._K_cells - self.individualsPerCell(), np.zeros(self._K_cells.shape)
            )
        else:
            available_space = np.maximum(
                self._K_species_cells - self._h, np.zeros(self._K_species_cells.shape)
            )


        if self._disturbance_dep_dispersal:
            self.update_dumping_dist(fast_dist)

        if self._dispersal_before_death == 0:
            if DEBUG:
                print("running NumCandidates, _dispersal_before_death")

            if skip_dispersal is False:
                if self._future_habitat_suitability is not None and update_suitability:
                    self.update_habitat_suitability()

                if DEBUG:
                    print("running NumCandidates, sij,ijnm->snm")
                # print("running NumCandidates, sij,ijnm->snm", self._h, self._dumping_dist)
                # NumCandidates = sparse.einsum(
                #     "sij,ijnm->snm", self._h, self._dumping_dist
                # )  # * self._growth_rate

                # NumCandidates = np.array(
                #     [sparse.einsum("ij,ijnm->nm",
                #                    self._h[i],
                #                    self._dumping_dist).todense() for i in range(self._n_species)])

                # growth
                # self._h = np.einsum("sij,s->sij", self._h, self._growth_rate)

                if not isinstance(self._lambda_0, Iterable):
                    NumCandidates = sparse.tensordot(self._h, self._dumping_dist ** (1 / self._lambda_0))
                    if DEBUG:
                        print("\nNumCandidates", self._lambda_0, NumCandidates[0][:10])
                else:
                    if DEBUG:
                        print("running NumCandidates, sij,ijnm->snm | per species dispersal rate")
                    NumCandidates = np.array(
                        [sparse.einsum("ij,ijnm->nm", # this is already the total number after migration
                                       self._h[i],
                                       self._dumping_dist ** (1 / self._lambda_0[i])
                                       ).todense() for i in range(self._n_species)])


                if self._K_species is not None:
                    # n_K = np.maximum(0, 1 - self._h / np.maximum(1, self.K_species3D))
                    # n_K = np.maximum(0, 1 - self._h / np.maximum(1, self._K_species * (
                    #         1 - np.einsum('s, xy -> sxy',
                    #                       self._disturbance_sensitivity,
                    #                       self.disturbance_matrix * (1 - self._protection_matrix)))))


                    # updated carrying capacity (w disturbance)
                    tmp_k = self.K_species3D * (1 - self.alphaHistogram(
                        self._disturbance_sensitivity,
                        self._disturbance_matrix * (1 - self._protection_matrix)))

                    if DEBUG:
                        print("\ntmp_k", np.sum(tmp_k[0]))
                        print("\nh", np.sum(self._h[0]))
                    # after adding migrants add growth (assumes growth rate > 1)
                    NumCandidates = np.einsum("sij,s->sij", NumCandidates, self._growth_rate)
                    if DEBUG:
                        print("\nNumCandidates0",np.sum(NumCandidates[0]))
                        print("\nnet NumCandidates0", np.sum(NumCandidates[0] - self.h[0]))
                    NumCandidates = np.minimum(
                        # here NumCandidates becomes the difference to current h
                        NumCandidates - self._h, tmp_k - self._h
                    )
                    NumCandidates[NumCandidates < 0] = 0
                    if DEBUG:
                        print("\nNumCandidates1", NumCandidates[0], (tmp_k - self._h)[0])

                    #
                    # n_K = np.maximum(0, 1 - self._h / np.maximum(1, tmp_k))
                    #
                    # growth_rate_3D =  np.einsum("sij,s->sij", n_K, self._growth_rate)
                    # NumCandidates = (1 + growth_rate_3D) * (NumCandidates + self._h)
                    # NumCandidates[self.K_species3D == 0] = 0
                    # print("\nNumCandidates", np.sum(NumCandidates[7]), np.mean(n_K[7]), np.sum(self._h[7]))
                    # i=0
                    # print("n_K:", n_K[i])
                    # print("growth_rate_3D[i]", growth_rate_3D[i])
                    # print("NumCandidates[i]", NumCandidates[i])
                else:
                    if DEBUG:
                        print("running NumCandidates, sij,s->sij")
                    NumCandidates = np.einsum("sij,s->sij", NumCandidates, self._growth_rate)

                if self._species_cell_specific_capacity is None:
                    totCandidates = np.einsum("sij->ij", NumCandidates)
                    normCandidates = NumCandidates / (totCandidates + small_number)
                else:
                    totCandidates = NumCandidates + 0
                    normCandidates = np.ones(NumCandidates.shape)
            else:
                totCandidates = np.zeros(self._h.shape)
                normCandidates = np.ones(self._h.shape)
                NumCandidates = np.zeros(self._h.shape)

        # print(NumCandidates)
        # print(np.einsum('sij,s->sij', NumCandidates,np.random.random(300)))
        # quit()
        # print(normCandidates.shape)
        # print(np.sum(normCandidates,axis=0))
        # print(np.einsum('sij->ij', NumCandidates))
        # quit()

        # replace individuals
        # dNdt = self._growth_rate * self.individualsPerCell() * (available_space/self._K_cells)
        # self._h = self._h + dNdt * normCandidates
        # self._h = self._h + available_space * normCandidates

        if DEBUG:
            print("running self._h update")
        if self._K_species is not None:
            if DEBUG:
                print("NumCandidates", NumCandidates[0])
            self._h = (self._h + NumCandidates) * (1 + self._alpha)
        else:
            tot_replaced = np.minimum(available_space, totCandidates)
            self._h = self._h + tot_replaced * normCandidates
            # print("available_space",available_space)

        # apply habitat suitability (must happen after dispersal)
        if self._habitat_suitability is not None and self._K_species is None:
            # kill individuals where they can't live
            self._h *= self._habitat_suitability

        # if self._K_species is not None:
        #     self._h = np.minimum(self._h, self._K_species)


        if self._truncateToInt:
            self._h = np.rint(self._h)
        # print(self._counter)

        if self._rm_lingering_pops:
            self._h[self._h < 1] = 0

        # self.

        self._counter += 1

    def set_species_values(self, v):
        "This is only used for plotting, nothing else"
        self._species_value_reference = v + 0

    def species_cell_specific_capacity(self, v):
        self._species_cell_specific_capacity = v + 0

    def set_dumping_dist(self, new_dist):
        self._dumping_dist = new_dist

    def set_reference_grid_pu(self, g):
        self._reference_grid_pu = g
        if g is not None:
            self._n_pus = g[g > 0].size

    def set_rm_lingering_pops(self, v):
        self._rm_lingering_pops = v

    @property
    def K_species3D(self):
        if self._K_species3D is None:
            self.update_K_species3D()

        return self._K_species3D

    @property
    def length(self):
        return self._length

    @property
    def h(self):
        return self._h

    @property
    def future_h(self):
        return self._future_habitat_suitability

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
