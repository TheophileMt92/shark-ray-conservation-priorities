import sys
import numpy as np
import scipy
from ..agents import state_monitor as state_monitor
from captain.plot.plot_features import plot_features
from captain.utilities import metrics

DEBUG = 0
class PolicyNN(object):
    def __init__(self,
                 num_features,
                 num_meta_features,
                 num_output,
                 coeff_features,
                 coeff_meta_features,
                 temperature,
                 mode,
                 observe_error,
                 use_true_natural_state=True,
                 nodes_l1=16,
                 nodes_l2=16,
                 nodes_l3=16,
                 sp_threshold=1,
                 fully_connected=0,
                 verbose=0,
                 feature_set=None,
                 flattened=False,
                 plot_features=False,
                 quadrant_coords_list=None,
                 wd_plot=None,
                 protection_constraint=None,
                 greedy_search=None,
                 ):
        # check meta features matches the features generating function
        self._num_meta_features = num_meta_features
        self._num_features = num_features
        self._num_output = num_output
        # init coefficients
        self._coeff = np.append(coeff_features, coeff_meta_features)
        self._temperature = temperature
        self._mode = mode  # indices of features
        self._observe_error = observe_error
        self._nodes_l1 = nodes_l1
        self._nodes_l2 = nodes_l2
        self._nodes_l3 = nodes_l3
        self._sp_threshold = sp_threshold
        self._fully_connected = fully_connected
        self._verbose = verbose
        self._flattened = flattened
        self._feature_set = feature_set
        self._plot_features = plot_features
        self._quadrant_coords_list = quadrant_coords_list
        self._wd_plot = wd_plot
        self._use_true_natural_state = use_true_natural_state
        self._lastObs = None
        self._protection_constraint = protection_constraint
        self._greedy_search = greedy_search
        if greedy_search is not None:
            self._greedy_search[1] = np.array(self._greedy_search[1])

    @property
    def num_output(self):
        return self._num_output

    @property
    def temperature(self):
        return self._temperature

    @property
    def coeff(self):
        return self._coeff

    def perturbeParams(self, noise):
        self._coeff += noise
        # self._coeff[ : -self._num_meta_features ] += noise[ : -self._num_meta_features ]
        # print("Before", self._coeff[self._num_features:])
        # self._coeff[self._num_features:] = UpdateUniform(self._coeff[ self._num_features : ] + 0)
        # print("after",self._coeff[self._num_features:])

    def setCoeff(self, newCoeff):
        self._coeff = newCoeff

    def probs(
        self, rich_state, lastObs=None, sp_quadrant_list_arg=None, return_lastObs=False
    ):

        # print(np.array(rich_state.protection_cost)[0:10])
        # logistic function applied here: coeff_meta_features must be between 0 and 1
        coeff_meta_features = state_monitor.get_thresholds(
            self._coeff[-self._num_meta_features :]
        )
        if lastObs is None:
            lastObs = state_monitor.extract_features(
                rich_state.grid_obj_most_recent,
                rich_state.grid_obj_previous,
                rich_state.resolution,
                rich_state.protection_matrix,
                rare_sp_qntl=coeff_meta_features[0],
                smallrange_sp_qntl=coeff_meta_features[self._num_meta_features - 1],
                mode=self._mode,
                observe_error=self._observe_error,
                cost_quadrant=rich_state.protection_cost,
                budget=rich_state.budget_left,
                sp_threshold=self._sp_threshold,
                sp_values=rich_state.sp_values,
                flattened=self._flattened,
                met_prot_target=rich_state.met_prot_target,
                min_pop_requirement=rich_state.min_pop_requirement,
                sp_quadrant_list_arg=sp_quadrant_list_arg,
            )

        state = lastObs.stats_quadrant
        if self._verbose:
            print(state[:20, :])
            print("features protected cells:")
            print(state[np.where(state[:, -1] == 1)[0], :])
            print(np.mean(state, 0), np.min(state, 0), np.max(state, 0))
        # if self._verbose:
        # print(lastObs.stats_quadrant[0:5,:])
        # print(np.sum(rich_state.grid_obj_most_recent.individualsPerSpecies()))

        # remove metafeatures
        coeff_policy = self._coeff[: -self._num_meta_features]
        internal_state = state[:, :]

        if self._fully_connected > 0:
            sys.exit("NN setting not available")
        elif self._nodes_l1 == 1:
            # linear regression
            h2 = np.einsum("nf, f -> n", internal_state, coeff_policy)
        else:  # NN with parameter sharing
            if self._nodes_l2:  # only used if using additional hidden layer
                tmp = coeff_policy[: self._num_features * self._nodes_l1]
                weights_l1 = tmp + 0
                tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                weights_l2 = coeff_policy[len(tmp) : -self._nodes_l3]
                # print(tmp_coeff.shape, weights_l2.shape)
                tmp_coeff2 = weights_l2.reshape(self._nodes_l1, self._nodes_l2)

                weights_l3 = coeff_policy[-self._nodes_l3 :]
            else:
                tmp = coeff_policy[: self._num_features * self._nodes_l1]
                weights_l1 = tmp + 0
                tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                weights_l3 = coeff_policy[-(self._nodes_l3) :]

            z1 = np.einsum("nf, fi->ni", internal_state, tmp_coeff)
            z1[z1 < 0] = 0
            if self._nodes_l2:
                # print(tmp_coeff2.shape, z1.shape)
                h1 = np.einsum("ni,ic->nc", z1, tmp_coeff2)
                h1[h1 < 0] = 0
                z1 = h1 + 0
            h2 = np.einsum("f,nf->n", weights_l3, z1)

        if self._temperature != 1:
            h2 *= self._temperature
            # same as
            """
            probs = scipy.special.softmax(h2)
            if self._temperature != 1:
                probs = probs ** self._temperature
                probs /= np.sum(probs)
            """
        # set to min prob probs of already protected units
        h2[internal_state[:, -1] == 1] = np.min(h2)
        probs = scipy.special.softmax(h2)
        # set to 0 probs of already protected units
        probs[internal_state[:, -1] == 1] = 0

        if np.sum(probs) < 1e-20:
            # avoid overflows
            probs += 1e-20
        probs /= np.sum(probs)
        if return_lastObs:
            return probs, lastObs
        else:
            return probs

    def setTemperature(self, temp):
        self._temperature = temp

    def reset_lastObs(self, lastObs):
        self._lastObs = lastObs



def get_NN_model_prm(num_features, n_NN_nodes, num_output):
    num_meta_features = 0  # TODO: check metafeatures
    nodes_layer_1 = n_NN_nodes[0]
    nodes_layer_2 = n_NN_nodes[1]  # set > 0 to add hidden layer
    if n_NN_nodes[0] == 1:
        nodes_layer_3 = 0
    elif n_NN_nodes[1] == 0:
        nodes_layer_3 = nodes_layer_1
    else:
        nodes_layer_3 = nodes_layer_2
    n_prms = (
        num_features * nodes_layer_1 + nodes_layer_1 * nodes_layer_2 + nodes_layer_3
    )
    # print("\n\nget_NN_model_prm")
    # print(num_features, nodes_layer_1, num_features * nodes_layer_1,nodes_layer_1 * nodes_layer_2, nodes_layer_3, n_prms)
    return [
        num_output,
        num_meta_features,
        nodes_layer_1,
        nodes_layer_2,
        nodes_layer_3,
        n_prms,
    ]


class PolicyRestore(PolicyNN):
    # def __init__(self,
    #              grid_sqrt_size,
    #              grid_h,
    #              env_resolution,
    #              ):
    #     r = state_monitor.get_quadrant_coord_species_clean(grid_sqrt_size, grid_h, env_resolution)
    #     self.quadrant_coords_list = r[0]

    def probs(self,
              rich_state,
              lastObs=None,
              sp_quadrant_list_arg=None,
              return_lastObs=False):

        # params
        if lastObs is None:
            lastObs = state_monitor.extract_features_restore(
                grid_obj=rich_state.grid_obj_most_recent,
                grid_obj_previous=rich_state.grid_obj_previous,
                quadrant_resolution=rich_state.resolution,
                current_protection_matrix=rich_state.protection_matrix,
                species_threat_label=rich_state.species_threat_label, # env.getExtinction_risk_labels()
                n_threat_labels=rich_state.n_threat_labels, # env.species_risk_criteria.n_labels
                quandrant_grid_indx=rich_state.quandrant_grid_indx, # env._quandrant_grid_indx
                cost_quadrant=rich_state.cost_quadrant, # env.getProtectCostQuadrant()
                mode=self._mode,
                feature_set=self._feature_set,
                observe_error=self._observe_error,
                use_true_natural_state=self._use_true_natural_state,
                budget=rich_state.budget_left,
                sp_threshold=self._sp_threshold,
                flattened=self._flattened,
                min_pop_requirement=rich_state.min_pop_requirement
            )

        state = lastObs.stats_quadrant

        if self._plot_features:
            plot_features(None, lastObs,
                          wd=self._wd_plot,
                          outfile="feat_%s" % rich_state.grid_obj_most_recent._counter,
                          quadrant_coords_list=self._quadrant_coords_list,
                          protection_matrix=rich_state.protection_matrix
                          )
            # print(hgfd)

        coeff_policy = self._coeff + 0
        internal_state = state[:, :]

        if self._nodes_l1 == 1:
            if self._nodes_l2 != -1:
                # linear regression
                h2 = np.einsum("nf, f -> n", internal_state, coeff_policy)
            else:
                h2 = np.zeros(internal_state.shape[0])
                h2[np.argmax(internal_state[:,0])] = 1
                if self._verbose:
                    print(internal_state[0, 0], np.argmax(internal_state[:,0]))
                    print(np.where(internal_state[:,0] == 0))
                    print(internal_state[:,0] )
                    print(h2)

        else:
            if self._nodes_l2:  # only used if using additional hidden layer
                    tmp = coeff_policy[: self._num_features * self._nodes_l1]
                    weights_l1 = tmp + 0
                    tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                    weights_l2 = coeff_policy[len(tmp) : -self._nodes_l3]
                    if DEBUG:
                        print(tmp_coeff.shape, weights_l2.shape)
                    tmp_coeff2 = weights_l2.reshape(self._nodes_l1, self._nodes_l2)

                    weights_l3 = coeff_policy[-self._nodes_l3 :]
            else:
                tmp = coeff_policy[: self._num_features * self._nodes_l1]
                weights_l1 = tmp + 0
                tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                weights_l3 = coeff_policy[-(self._nodes_l3) :]

            z1 = np.einsum("nf, fi->ni", internal_state, tmp_coeff)
            # z1[z1 < 0] = 0
            z1 = np.tanh(z1)
            if self._nodes_l2:
                # print(tmp_coeff2.shape, z1.shape)
                h1 = np.einsum("ni,ic->nc", z1, tmp_coeff2)
                # h1[h1 < 0] = 0
                # z1 = h1 + 0
                z1 = np.tanh(h1)
            h2 = np.einsum("f,nf->n", weights_l3, z1)

        if self._temperature != 1:
            h2 *= self._temperature

        # set to min prob probs of already protected units
        h2[lastObs.protected_quadrants == 1] = np.min(h2)
        probs = scipy.special.softmax(h2)
        # set to 0 probs of already protected units
        probs[lastObs.protected_quadrants == 1] = 0
        if np.sum(probs) < 1e-20:
            # avoid overflows
            probs += 1e-20
        probs /= np.sum(probs)
        if return_lastObs:
            return probs, lastObs
        else:
            return probs


class PolicyRestoreUpdateFreq(PolicyNN):
    # def __init__(self,
    #              grid_sqrt_size,
    #              grid_h,
    #              env_resolution,
    #              ):
    #     r = state_monitor.get_quadrant_coord_species_clean(grid_sqrt_size, grid_h, env_resolution)
    #     self.quadrant_coords_list = r[0]

    def probs(self,
              rich_state,
              lastObs=None,
              sp_quadrant_list_arg=None,
              return_lastObs=False):

        # params
        if self._lastObs is None:
            if self._verbose:
                print("\nExtracting features...")
            lastObs = state_monitor.extract_features_restore(
                grid_obj=rich_state.grid_obj_most_recent,
                grid_obj_previous=rich_state.grid_obj_previous,
                quadrant_resolution=rich_state.resolution,
                current_protection_matrix=rich_state.protection_matrix,
                species_threat_label=rich_state.species_threat_label, # env.getExtinction_risk_labels()
                n_threat_labels=rich_state.n_threat_labels, # env.species_risk_criteria.n_labels
                quandrant_grid_indx=rich_state.quandrant_grid_indx, # env._quandrant_grid_indx
                cost_quadrant=rich_state.cost_quadrant, # env.getProtectCostQuadrant()
                mode=self._mode,
                feature_set=self._feature_set,
                observe_error=self._observe_error,
                use_true_natural_state=self._use_true_natural_state,
                budget=rich_state.budget_left,
                sp_threshold=self._sp_threshold,
                flattened=self._flattened,
                min_pop_requirement=rich_state.min_pop_requirement,
                future_species_threat_label=rich_state.species_future_threat_label
            )
            self.reset_lastObs(lastObs)
            # print("reset_lastObs", self._lastObs)
        else:
            # update protected quadrants
            self._lastObs.protected_quadrants = state_monitor.get_protected_quadrants(
                protection_matrix=rich_state.protection_matrix,
                quandrant_grid_indx=rich_state.quandrant_grid_indx)

        state = self._lastObs.stats_quadrant
        # print("PolicyRestoreUpdateFreq PROBS: grid_obj_previous, grid_obj", np.sum(rich_state.grid_obj_previous.h), np.sum(rich_state.grid_obj_most_recent.h))

        if self._plot_features:
            plot_features(None, lastObs,
                          wd=self._wd_plot,
                          outfile="feat_%s" % rich_state.grid_obj_most_recent._counter,
                          quadrant_coords_list=self._quadrant_coords_list,
                          protection_matrix=rich_state.protection_matrix
                          )
            # print(hgfd)

        if self._greedy_search is not None:
            # print("greedy_search\n", state.shape, self._greedy_search, self._feature_set, self._lastObs.feature_names)
            # list: [0] <- names of features
            # list: [1] <- >0 to maximize, <0 to minimize (value = weight)
            cr_ind = np.array([np.where(self._lastObs.feature_names == i)[0][0] for i in self._greedy_search[0]])
            # print("ind", cr_ind, np.mean(state[:, cr_ind]), state[:, cr_ind], self._greedy_search)
            val = (1e-50 + state[:, cr_ind])
            # val_r_e = (val / np.maximum(np.mean(val, axis=0), 1e-50)) * self._greedy_search[1]

            # use ratio in log space

            val_r = val / np.maximum(np.mean(val, axis=0), 1e-50)
            log_num = np.log((val_r[:, self._greedy_search[1] > 0] * self._greedy_search[1][self._greedy_search[1] > 0]))

            log_den = np.log((val_r[:, self._greedy_search[1] < 0] * np.abs(self._greedy_search[1])[self._greedy_search[1] < 0]))

            val_r_e = log_num - log_den


            # set lowest values to protected cells
            val_r_e[rich_state.protection_matrix.flatten() == 1,:] = np.min(val_r_e)

            probs = scipy.special.softmax(np.sum(val_r_e, axis=1))

        else:
            # run NN
            coeff_policy = self._coeff + 0
            internal_state = state[:, :]

            if self._nodes_l1 == 1:
                if self._nodes_l2 != -1:
                    # linear regression
                    h2 = np.einsum("nf, f -> n", internal_state, coeff_policy)
                else:
                    h2 = np.zeros(internal_state.shape[0])
                    h2[np.argmax(internal_state[:,0])] = 1
                    if self._verbose:
                        print(internal_state[0, 0], np.argmax(internal_state[:,0]))
                        print(np.where(internal_state[:,0] == 0))
                        print(internal_state[:,0] )
                        print(h2)

            else:
                if self._nodes_l2:  # only used if using additional hidden layer
                        tmp = coeff_policy[: self._num_features * self._nodes_l1]
                        weights_l1 = tmp + 0
                        tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                        weights_l2 = coeff_policy[len(tmp) : -self._nodes_l3]
                        if DEBUG:
                            print(tmp_coeff.shape, weights_l2.shape)
                        tmp_coeff2 = weights_l2.reshape(self._nodes_l1, self._nodes_l2)

                        weights_l3 = coeff_policy[-self._nodes_l3 :]
                else:
                    tmp = coeff_policy[: self._num_features * self._nodes_l1]
                    weights_l1 = tmp + 0
                    tmp_coeff = weights_l1.reshape(self._num_features, self._nodes_l1)

                    weights_l3 = coeff_policy[-(self._nodes_l3) :]

                z1 = np.einsum("nf, fi->ni", internal_state, tmp_coeff)
                # z1[z1 < 0] = 0
                z1 = np.tanh(z1)
                if self._nodes_l2:
                    # print(tmp_coeff2.shape, z1.shape)
                    h1 = np.einsum("ni,ic->nc", z1, tmp_coeff2)
                    # h1[h1 < 0] = 0
                    # z1 = h1 + 0
                    z1 = np.tanh(h1)
                h2 = np.einsum("f,nf->n", weights_l3, z1)

            if self._temperature != 1:
                h2 *= self._temperature

            # set to min prob probs of already protected units
            h2[self._lastObs.protected_quadrants == 1] = np.min(h2)
            probs = scipy.special.softmax(h2)

        # set to 0 probs of already protected units
        probs[self._lastObs.protected_quadrants == 1] = 0
        if np.sum(probs) < 1e-20:
            # avoid overflows
            probs += 1e-20
        probs /= np.sum(probs)

        if self._protection_constraint is not None:
            # print("USING _protection_constraint")
            probs_tmp = self._protection_constraint + 0
            probs_tmp[probs_tmp < 0] = 0
            # print(#np.unique(rich_state.protection_matrix.flatten()),
            #     np.sum(self._protection_constraint),
            #     np.sum(probs_tmp),
            #     np.sum(rich_state.protection_matrix),
            #     np.sum(self._lastObs.protected_quadrants)
            # )
            # print(np.unique(probs_tmp))
            if np.max(probs_tmp) < 1:
                self._protection_constraint = None
                # print("STOPPING _protection_constraint")
            else:
                probs_tmp_r = probs_tmp + probs # add to other vec
                probs_tmp_r[self._lastObs.protected_quadrants == 1] = 0
                probs_tmp_r += 1e-20
                probs = probs_tmp_r / np.sum(probs_tmp_r)
                # probs[probs_tmp > 0] = probs_tmp[probs_tmp > 0] + 0
                # probs /= np.sum(probs)
            # print(np.sort(probs)[::-1][:5], np.max(probs), np.min(probs))

        if return_lastObs:
            return probs, lastObs
        else:
            return probs



class PolicyHeuristicUpdateFreq(PolicyNN):
    # def __init__(self,
    #              grid_sqrt_size,
    #              grid_h,
    #              env_resolution,
    #              ):
    #     r = state_monitor.get_quadrant_coord_species_clean(grid_sqrt_size, grid_h, env_resolution)
    #     self.quadrant_coords_list = r[0]

    def set_heuristic_policy(self, heuristic_policy, minimize_policy):
        self.heuristic_policy = heuristic_policy
        self.minimize_policy = minimize_policy
    def probs(self,
              rich_state,
              lastObs=None,
              sp_quadrant_list_arg=None,
              return_lastObs=False):

        # params
        if self._lastObs is None:
            print("\nExtracting features...")
            lastObs = state_monitor.extract_features_restore(
                grid_obj=rich_state.grid_obj_most_recent,
                grid_obj_previous=rich_state.grid_obj_previous,
                quadrant_resolution=rich_state.resolution,
                current_protection_matrix=rich_state.protection_matrix,
                species_threat_label=rich_state.species_threat_label, # env.getExtinction_risk_labels()
                n_threat_labels=rich_state.n_threat_labels, # env.species_risk_criteria.n_labels
                quandrant_grid_indx=rich_state.quandrant_grid_indx, # env._quandrant_grid_indx
                cost_quadrant=rich_state.cost_quadrant, # env.getProtectCostQuadrant()
                mode=self._mode,
                feature_set=self._feature_set,
                observe_error=self._observe_error,
                use_true_natural_state=self._use_true_natural_state,
                budget=rich_state.budget_left,
                sp_threshold=self._sp_threshold,
                flattened=self._flattened,
                min_pop_requirement=rich_state.min_pop_requirement
            )
            self.reset_lastObs(lastObs)
            # print("reset_lastObs", self._lastObs)
        else:
            # update protected quadrants
            self._lastObs.protected_quadrants = state_monitor.get_protected_quadrants(
                protection_matrix=rich_state.protection_matrix,
                quandrant_grid_indx=rich_state.quandrant_grid_indx)

        # state = self._lastObs.stats_quadrant

        if self._plot_features:
            plot_features(None, lastObs,
                          wd=self._wd_plot,
                          outfile="feat_%s" % rich_state.grid_obj_most_recent._counter,
                          quadrant_coords_list=self._quadrant_coords_list,
                          protection_matrix=rich_state.protection_matrix
                          )
            # print(hgfd)


        # print("self.heuristic_policy", self.heuristic_policy)
        if self.heuristic_policy == "random":
            h2 = np.ones(rich_state.cost_quadrant.shape)
        elif self.heuristic_policy == "cheapest":
            costs = rich_state.cost_quadrant
            h2 = 1 / (costs + np.min(costs[costs > 0]))
        elif self.heuristic_policy == "most_biodiverse":
            h2 = state_monitor.get_mean_grid_value_quadrant(rich_state.grid_obj_most_recent.speciesPerCell(),
                                                            quandrant_grid_indx=rich_state.quandrant_grid_indx)
        elif self.heuristic_policy == "most_natural_biodiversity":
            h2 = state_monitor.get_mean_grid_value_quadrant(rich_state.grid_obj_previous.speciesPerCell(),
                                                            quandrant_grid_indx=rich_state.quandrant_grid_indx)
        elif self.heuristic_policy == "most_natural_carbon":
            h2 = state_monitor.get_mean_grid_value_quadrant(rich_state.grid_obj_previous.getCarbonValue_cell(),
                                                            quandrant_grid_indx=rich_state.quandrant_grid_indx)
        elif self.heuristic_policy == "most_current_carbon":
            h2 = state_monitor.get_mean_grid_value_quadrant(rich_state.grid_obj_most_recent.getCarbonValue_cell(),
                                                            quandrant_grid_indx=rich_state.quandrant_grid_indx)
        elif self.heuristic_policy == "highest_MSA":
            h2 = metrics.calc_MSA_from_grid(rich_state.grid_obj_previous.h, rich_state.grid_obj_most_recent.h,
                                            quandrant_grid_indx=rich_state.quandrant_grid_indx)
        elif self.heuristic_policy == "highest_STAR_t":
            h2, _ = metrics.calc_STAR_from_grid(rich_state.grid_obj_previous.h, rich_state.grid_obj_most_recent.h,
                                                quandrant_grid_indx=rich_state.quandrant_grid_indx,
                                                sp_natural_range=rich_state.grid_obj_previous.geoRangePerSpecies(),
                                                sp_ext_risk=rich_state.species_threat_label)
        elif self.heuristic_policy == "highest_STAR_r":
            _, h2 = metrics.calc_STAR_from_grid(rich_state.grid_obj_previous.h, rich_state.grid_obj_most_recent.h,
                                                quandrant_grid_indx=rich_state.quandrant_grid_indx,
                                                sp_natural_range=rich_state.grid_obj_previous.geoRangePerSpecies(),
                                                sp_ext_risk=rich_state.species_threat_label)

        self._probs_heuristic_policy = h2 + 0


        # apply minimization policy
        if self.minimize_policy:
            h2 = 1 / (h2 + 0.00001)

        # remove e.g. sea
        m = state_monitor.get_mean_grid_value_quadrant(rich_state.grid_obj_most_recent.bioDivGrid._K_cells > 0,
                                         quandrant_grid_indx=rich_state.quandrant_grid_indx)
        h2[m == 0] = 0


        if self._temperature != 1:
            h2 *= self._temperature

        # set to min prob probs of already protected units
        h2[self._lastObs.protected_quadrants == 1] = np.min(h2)
        probs = scipy.special.softmax(h2)
        # set to 0 probs of already protected units
        probs[self._lastObs.protected_quadrants == 1] = 0
        if np.sum(probs) < 1e-20:
            # avoid overflows
            probs += 1e-20
        probs /= np.sum(probs)
        if return_lastObs:
            return probs, lastObs
        else:
            return probs