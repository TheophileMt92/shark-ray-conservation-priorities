import numpy as np
np.set_printoptions(suppress=True, precision=3)


class genericRewardClass:
    def __init__(self, verbose=False):
        self.verbose = verbose


    def get_reward(self, species_grid):
        return np.mean(species_grid)


class mltRewardPareto(genericRewardClass):
    def __init__(self, reward_weights=None, verbose=False):
        super().__init__(verbose)
        self._reward_weights = reward_weights


    def calc_reward(self,
                    carbon_cell,
                    init_total_carbon,
                    total_natural_carbon,
                    cumrewards,
                    rl):
        # carbon
        tot_reward = np.sum(carbon_cell) - init_total_carbon
        tot_reward = tot_reward / (total_natural_carbon - init_total_carbon) * 100
        reward_c = (tot_reward * self._reward_weights['carbon'] - cumrewards[0])

        # species rl
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
        reward_sp = (tot_reward_species * self._reward_weights['species_risk'] - self._cumrewards[1])

