from nsga2.utils import NSGA2Utils
from nsga2.population import Population
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import random
import numpy as np

class Evolution:

    def __init__(self, problem, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.3, beta_method="uniform", patience=None):
        self.utils = NSGA2Utils(problem, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)
        self.population = None
        self.evolutions_df = pd.DataFrame()
        self.dataset = problem.dataset
        self.sensitive_attribute = problem.sensitive_attribute
        self.num_of_generations = problem.generations
        self.on_generation_finished = []
        self.num_of_individuals = problem.individuals
        self.mutation_prob = mutation_prob
        self.beta_method = beta_method
        self.patience = patience

    def evolve(self):
        #random.seed(self.utils.problem.seed)  
        #np.random.seed(self.utils.problem.seed)
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        best_eff = np.inf
        best_fair = np.inf
        for i in tqdm(range(self.num_of_generations)):
            for indiv in self.population.population:
                indiv_list = list(indiv.features.items())
                if self.utils.problem.classifier == "tree":
                    criterion, max_depth, min_samples_split, max_leaf_nodes = [item[1] for item in indiv_list]
                    evolutions_aux = pd.DataFrame({'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes],  'eff_metric': indiv.objectives[0], 'fair_metric': indiv.objectives[1], 'generation': i+1, 'rank': indiv.rank, 'actual_depth': indiv.actual_depth, 'actual_leaves': indiv.actual_leaves, 'id': indiv.id, 'creation_mode': indiv.creation_mode, 'seed': self.utils.problem.seed})
                elif self.utils.problem.classifier == "logit":
                    max_iter, tol, C, ratio = [item[1] for item in indiv_list]
                    evolutions_aux = pd.DataFrame({'max_iter': [max_iter], 'tol': [tol], 'C': [C], 'ratio': [ratio],  'eff_metric': indiv.objectives[0], 'fair_metric': indiv.objectives[1], 'generation': i+1, 'rank': indiv.rank,  'id': indiv.id, 'creation_mode': indiv.creation_mode, 'seed': self.utils.problem.seed})
                elif self.utils.problem.classifier == "rf":
                    n_estimators, max_features, max_samples, criterion, max_depth, min_samples_split, max_leaf_nodes,  = [item[1] for item in indiv_list]
                    evolutions_aux = pd.DataFrame({'n_estimators': [n_estimators], 'max_features': [max_features], 'max_samples': [max_samples], 'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes],  'eff_metric': indiv.objectives[0], 'fair_metric': indiv.objectives[1], 'generation': i+1, 'rank': indiv.rank, 'id': indiv.id, 'creation_mode': indiv.creation_mode, 'seed': self.utils.problem.seed})
                
                self.evolutions_df = pd.concat([self.evolutions_df, evolutions_aux])
            
            gen_eff = np.mean(evolutions_aux["eff_metric"])
            gen_fair = np.mean(evolutions_aux["fair_metric"])

            if gen_eff < best_eff:
                best_eff = gen_eff
                self.num_of_generations_without_improvement = 0
            elif gen_fair < best_fair:
                best_fair = gen_fair
                self.num_of_generations_without_improvement = 0
            else:
                self.num_of_generations_without_improvement += 1

            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)

            if self.num_of_generations_without_improvement >= self.patience:
                print(f"Stopping criterion met: Pareto front did not improve in {self.patience} generations.")
                break

            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals-len(new_population)])
            returned_population = self.population
            self.population = new_population
            children = self.utils.create_children(self.population)
            

        return returned_population.fronts[0]
