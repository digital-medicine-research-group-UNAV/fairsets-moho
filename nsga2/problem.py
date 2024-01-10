import random
import string
import copy
from collections import OrderedDict as od
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from nsga2.individual import Individual
from nsga2.toolbox import *
from collections import OrderedDict as od


class Problem:

    def __init__(
            self,
            config,
        ):
        
        self.config = config
        self.dataset = config["dataset"]
        self.classifier = config["classifier"]
        self.eff_obj = config["eff_obj"]
        self.sensitive_groups = config["sensitive_groups"]
        self.fair_obj = config["fair_obj"]
        self.expand = config["expand"]
        self.variables_range = config[config["classifier"] + "_space"]
        self.individuals_df = pd.DataFrame()
        self.generations = config["generations"]
        self.individuals = config["individuals"]
        self.label = config["label"]
        self.sensitive_attribute = config["sensitive_attribute"]
        self.priviledged_value = config["priv_value"]
        self.confidence = config["conf"]
        self.cal_size = config["cal_size"]
        self.seed = config["set_seed"]

    def create_datasets(self):

        df = pd.read_csv(f"data/clean/{self.dataset}.csv")

        # If no priviledged value is provided, chose majority group
        if self.priviledged_value == None:
            self.priviledged_value = df[self.sensitive_attribute].value_counts().idxmax()
        
        # If sensitive attribute is binarized, all the unpriviledged groups are bucketed
        if self.config["sensitive_groups"] == "bin":
            df[self.sensitive_attribute] = df[self.sensitive_attribute].apply(lambda x: 0 if x == self.priviledged_value else 1)
            self.priviledged_value = 0

        # Feature, sensitive attribute and target  
        X = df.drop([self.label], axis=1)
        sens = df[self.sensitive_attribute]
        y = df[self.label]

        self.X_learn, self.X_cal, self.X_val, self.X_test, self.sens_learn, self.sens_cal, self.sens_val, self.sens_test, self.y_learn, self.y_cal, self.y_val, self.y_test = \
            train_cal_val_test_split(X, sens, y,  sizes=[0.5-(self.cal_size/2), self.cal_size/2, 0.25, 0.25])
        
        if self.classifier == "tree":
            self.variables_range["min_samples_split"][1] = len(self.y_learn)/10
        if self.classifier == "rf":
            self.variables_range["min_samples_split"][1] = len(self.y_learn)/10

        return self

    def generate_individual(self, order=None):       
        individual = Individual()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        if order == "first":
            individual.features = self.config["initial_" + self.classifier]["first"]
        elif order == "second":
            individual.features = self.config["initial_" + self.classifier]["second"]
        else:
            individual.features = [random.uniform(*x) for x in self.variables_range.values()]
        
        hyperparameters = self.config[self.classifier + "_hyps"]      
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.classifier, self.variables_range, **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    def calculate_objectives(self, individual, first_individual, seed):
        if self.expand:
            hyperparameters = individual.features
            
            # Train classifier
            if self.classifier == "tree":
                learner = learn_decision_tree(self.X_learn, self.y_learn, **hyperparameters)
            if self.classifier == "logit":
                learner = learn_logit(self.X_learn, self.y_learn, **hyperparameters)
            if self.classifier == "rf":
                learner = learn_random_forest(self.X_learn, self.y_learn, **hyperparameters)
            
            # Calibrate model using crepes and define an inductive conformal predictor
            if self.classifier == "rf":
                conformal_predictor = calibrate_random_forest(learner, self.X_cal, self.y_cal)
            else:
                conformal_predictor = calibrate_classifier(learner, self.X_cal, self.y_cal)

            # Compute coverage on validation set
            coverage = marginal_coverage(conformal_predictor, self.X_val, self.y_val, self.confidence)

            # Compute per-group coverage on validation set
            cov_dict = group_specific_coverage(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence)

            # Compute efficiency on validation set
            if self.eff_obj == "set_size":
                eff_metric = average_set_size(conformal_predictor, self.X_val, self.y_val, self.confidence)

            # Compute per-group efficiency on validation set
            eff_dict = group_specific_efficiency(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence)

            # Compute unfairness on validation set
            if self.fair_obj == "theoretical_cov_gap":
                fair_metric = 100 * np.mean(np.abs(np.array(list(cov_dict.values())) - (self.confidence)))
            elif self.fair_obj == "avg_priv_dif":
                fair_metric = 100 * avg_priv_dif(cov_dict, self.priviledged_value)
            elif self.fair_obj == "max_priv_dif":
                fair_metric = 100 * max_priv_dif(cov_dict, self.priviledged_value)


            cov_dict["marg_cov"] = coverage
            individual.objectives = [eff_metric, fair_metric]

            if eff_metric == self.y_learn.nunique():
                individual.objectives[1] = 100

            parameters = {key: [value] for key, value in individual.features.items()}
            cov_dict = {key: [value] for key, value in cov_dict.items()}
            eff_dict = {key: [value] for key, value in eff_dict.items()}

            if self.classifier == "tree":
                individual.actual_depth = learner.get_depth()
                individual.actual_leaves = learner.get_n_leaves()
                if first_individual:
                    self.config["initial_tree"]["second"][1] = individual.actual_depth
                    self.config["initial_tree"]["second"][3] = individual.actual_leaves
                    self.variables_range["max_depth"][1] = individual.actual_depth
                    self.variables_range["max_leaf_nodes"][1] = individual.actual_leaves
                parameters["actual_depth"] = individual.actual_depth
                parameters["actual_leaves"] = individual.actual_leaves

            if self.classifier == "rf":
                rf_depths = [tree.get_depth() for tree in learner.estimators_]
                individual.avg_depth = sum(rf_depths) / len(rf_depths)
                rf_leaves = [tree.get_n_leaves() for tree in learner.estimators_]
                individual.avg_leaves = sum(rf_leaves) / len(rf_leaves)                        
                if(first_individual):
                    self.config["initial_rf"]["second"][4] = individual.avg_depth
                    self.config["initial_rf"]["second"][6] = individual.avg_leaves
                    self.variables_range["max_depth"][1] = individual.avg_depth
                    self.variables_range["max_leaf_nodes"][1] = individual.avg_leaves                        
                parameters["avg_depth"] = individual.avg_depth
                parameters["avg_leaves"] = individual.avg_leaves

            individual_info = {'id': individual.id, 'creation_mode':individual.creation_mode, self.eff_obj: eff_metric, "unfairness": fair_metric}
            individual_info.update(parameters)
            individual_info.update(cov_dict)
            individual_info.update(eff_dict)

            individuals_aux = pd.DataFrame(individual_info)
            self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            
            self.individuals_df.to_csv(
                f'./results/{self.dataset}/{self.fair_obj}/{self.classifier}/individuals/individuals_{self.sensitive_groups}{self.sensitive_attribute}_calsize_{str(self.cal_size)[2:]}_seed_{str(seed)}_gen_{str(self.generations)}_indiv_{str(self.individuals)}.csv',
                index = False,
                columns = list(individual_info.keys()) 
            )

    def test_and_save(self, individual, first, seed):
        if self.expand:
            hyperparameters = individual.features
            
            # Train classifier
            if self.classifier == "tree":
                learner = learn_decision_tree(self.X_learn, self.y_learn, **hyperparameters)
            elif self.classifier == "logit":
                learner = learn_logit(self.X_learn, self.y_learn, **hyperparameters)
            elif self.classifier == "rf":
                learner = learn_random_forest(self.X_learn, self.y_learn, **hyperparameters)
            
            # Calibrate model using crepes and define an inductive conformal predictor
            if self.classifier == "rf":
                conformal_predictor = calibrate_random_forest(learner, self.X_cal, self.y_cal)
            else:
                conformal_predictor = calibrate_classifier(learner, self.X_cal, self.y_cal)

            # Compute marginal coverage
            coverage_val = marginal_coverage(conformal_predictor, self.X_val, self.y_val, self.confidence)
            coverage_test = marginal_coverage(conformal_predictor, self.X_test, self.y_test, self.confidence)
            
            # Compute per-group coverage
            cov_dict_val = group_specific_coverage(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence)
            cov_dict_test = group_specific_coverage(conformal_predictor, self.X_test, self.y_test, self.sens_test, self.confidence)
            
            # Compute per-group efficiency
            eff_dict_val = group_specific_efficiency(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence)
            eff_dict_test = group_specific_efficiency(conformal_predictor, self.X_test, self.y_test, self.sens_test, self.confidence)

            # Compute efficiency
            if self.eff_obj == "set_size":
                eff_metric_val = average_set_size(conformal_predictor, self.X_val, self.y_val, self.confidence)
                eff_metric_test = average_set_size(conformal_predictor, self.X_test, self.y_test, self.confidence)

            # Compute unfairness on test set
            if self.fair_obj == "theoretical_cov_gap":
                fair_metric_val = 100 * np.mean(np.abs(np.array(list(cov_dict_val.values())) - (self.confidence)))
                fair_metric_test = 100 * np.mean(np.abs(np.array(list(cov_dict_test.values())) - (self.confidence)))
            elif self.fair_obj == "avg_priv_dif":
                fair_metric_val = 100 * avg_priv_dif(cov_dict_val, self.priviledged_value)
                fair_metric_test = 100 * avg_priv_dif(cov_dict_test, self.priviledged_value)
            elif self.fair_obj == "max_priv_dif":
                fair_metric_val = 100 * max_priv_dif(cov_dict_val, self.priviledged_value)
                fair_metric_test = 100 * max_priv_dif(cov_dict_val, self.priviledged_value)

            cov_dict_val = {key + '_val': [value] for key, value in cov_dict_val.items()}
            cov_dict_test = {key + '_test': [value] for key, value in cov_dict_test.items()}
            eff_dict_val = {key + '_val': [value] for key, value in eff_dict_val.items()}
            eff_dict_test = {key + '_test': [value] for key, value in eff_dict_test.items()}

            cov_dict_val["marg_cov_val"] = [coverage_val]
            cov_dict_test["marg_cov_test"] = [coverage_test]

            parameters = {key: [value] for key, value in individual.features.items()}
            
            individual_info = {'id': individual.id, 'creation_mode': "mo_opt", self.eff_obj + "_val": eff_metric_val, "unfairness_val": fair_metric_val, self.eff_obj + "_test": eff_metric_test, "unfairness_test": fair_metric_test}
            individual_info.update(parameters)
            individual_info.update(cov_dict_val)
            individual_info.update(cov_dict_test)
            individual_info.update(eff_dict_val)
            individual_info.update(eff_dict_test)

            individuals_aux = pd.DataFrame(individual_info)
            self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            
            self.individuals_df.to_csv(
                f'./results/{self.dataset}/{self.fair_obj}/{self.classifier}/individuals/individuals_pareto_{self.sensitive_groups}{self.sensitive_attribute}_calsize_{str(self.cal_size)[2:]}_seed_{str(seed)}_gen_{str(self.generations)}_indiv_{str(self.individuals)}.csv',
                index = False,
                columns = list(individual_info.keys())
            )
            
    def benchmark_conformal(self, seed):
        """
        Perform benchmark experiment using standard CP for comparison purposes.
        """
        # Train classifier
        if self.classifier == "tree":
            learner = learn_default_decision_tree(self.X_learn, self.y_learn)
        elif self.classifier == "logit":
            learner = learn_default_logit(self.X_learn, self.y_learn)
        elif self.classifier == "rf":
            learner = learn_default_random_forest(self.X_learn, self.y_learn)
            
        # Calibrate model using crepes and define an inductive conformal predictor
        if self.classifier == "rf":
            conformal_predictor = calibrate_random_forest(learner, self.X_cal, self.y_cal)
        else:
            conformal_predictor = calibrate_classifier(learner, self.X_cal, self.y_cal)

        # Compute marginal coverage
        coverage_val = marginal_coverage(conformal_predictor, self.X_val, self.y_val, self.confidence)
        coverage_test = marginal_coverage(conformal_predictor, self.X_test, self.y_test, self.confidence)
            
        # Compute per-group coverage
        cov_dict_val = group_specific_coverage(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence)
        cov_dict_test = group_specific_coverage(conformal_predictor, self.X_test, self.y_test, self.sens_test, self.confidence)
            
        # Compute per-group efficiency
        eff_dict_val = group_specific_efficiency(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence)
        eff_dict_test = group_specific_efficiency(conformal_predictor, self.X_test, self.y_test, self.sens_test, self.confidence)

        # Compute efficiency on validation set
        if self.eff_obj == "set_size":
            eff_metric_val = average_set_size(conformal_predictor, self.X_val, self.y_val, self.confidence)

        # Compute efficiency on test set
        if self.eff_obj == "set_size":
            eff_metric_test = average_set_size(conformal_predictor, self.X_test, self.y_test, self.confidence)

        # Compute unfairness on validaton set
        if self.fair_obj == "theoretical_cov_gap":
            fair_metric_val = 100 * np.mean(np.abs(np.array(list(cov_dict_val.values())) - (self.confidence)))
        elif self.fair_obj == "avg_priv_dif":
            fair_metric_val = 100 * avg_priv_dif(cov_dict_val, self.priviledged_value)
        elif self.fair_obj == "max_priv_dif":
            fair_metric_val = 100 * max_priv_dif(cov_dict_val, self.priviledged_value)

        # Compute unfairness on test set
        if self.fair_obj == "theoretical_cov_gap":
            fair_metric_test = np.mean(np.abs(np.array(list(cov_dict_test.values())) - (self.confidence)))
        elif self.fair_obj == "avg_priv_dif":
            fair_metric_test = 100 * avg_priv_dif(cov_dict_test, self.priviledged_value)
        elif self.fair_obj == "max_priv_dif":
            fair_metric_test = 100 * max_priv_dif(cov_dict_test, self.priviledged_value)

        cov_dict_val = {key + '_val': [value] for key, value in cov_dict_val.items()}
        cov_dict_test = {key + '_test': [value] for key, value in cov_dict_test.items()}
        eff_dict_val = {key + '_val': [value] for key, value in eff_dict_val.items()}
        eff_dict_test = {key + '_test': [value] for key, value in eff_dict_test.items()}

        cov_dict_val["marg_cov_val"] = [coverage_val]
        cov_dict_test["marg_cov_test"] = [coverage_test]

        parameters = {}
        for param in self.config[self.config["classifier"] + "_hyps"]:
            parameters[param] = learner.get_params()[param]

        individual_info = {'id': "", 'creation_mode': "icp", self.eff_obj + "_val": eff_metric_val, "unfairness_val": fair_metric_val, self.eff_obj + "_test": eff_metric_test, "unfairness_test": fair_metric_test}
        individual_info.update(parameters)
        individual_info.update(cov_dict_val)
        individual_info.update(cov_dict_test)
        individual_info.update(eff_dict_val)
        individual_info.update(eff_dict_test)

        individuals_aux = pd.DataFrame(individual_info)
        self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            
        self.individuals_df.to_csv(
            f'./results/{self.dataset}/{self.fair_obj}/{self.classifier}/individuals/individuals_pareto_{self.sensitive_groups}{self.sensitive_attribute}_calsize_{str(self.cal_size)[2:]}_seed_{str(seed)}_gen_{str(self.generations)}_indiv_{str(self.individuals)}.csv',
            index = False,
            columns = list(individual_info.keys())
        ) 
    
    def benchmark_mondrian_conformal(self, seed):
        """
        Perform benchmark experiment using Mondrian CP for comparison purposes.
        """
        # Train classifier
        if self.classifier == "tree":
            learner = learn_default_decision_tree(self.X_learn, self.y_learn)
        elif self.classifier == "logit":
            learner = learn_default_logit(self.X_learn, self.y_learn)
        elif self.classifier == "rf":
            learner = learn_default_random_forest(self.X_learn, self.y_learn)
            
        # Calibrate model using crepes and define an inductive conformal predictor
        if self.classifier == "rf":
            conformal_predictor = calibrate_random_forest(learner, self.X_cal, self.y_cal, taxonomy=self.sens_cal)
        else:
            conformal_predictor = calibrate_classifier(learner, self.X_cal, self.y_cal, taxonomy=self.sens_cal)

        # Compute marginal coverage
        coverage_val = marginal_coverage(conformal_predictor, self.X_val, self.y_val, self.confidence, taxonomy=self.sens_val)
        coverage_test = marginal_coverage(conformal_predictor, self.X_test, self.y_test, self.confidence, taxonomy=self.sens_test)
            
        # Compute per-group coverage
        cov_dict_val = group_specific_coverage(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence, taxonomy=self.sens_val)
        cov_dict_test = group_specific_coverage(conformal_predictor, self.X_test, self.y_test, self.sens_test, self.confidence, taxonomy=self.sens_test)
            
        # Compute per-group efficiency
        eff_dict_val = group_specific_efficiency(conformal_predictor, self.X_val, self.y_val, self.sens_val, self.confidence, taxonomy=self.sens_val)
        eff_dict_test = group_specific_efficiency(conformal_predictor, self.X_test, self.y_test, self.sens_test, self.confidence, taxonomy=self.sens_test)

        # Compute efficiency on validation set
        if self.eff_obj == "set_size":
            eff_metric_val = average_set_size(conformal_predictor, self.X_val, self.y_val, self.confidence, taxonomy=self.sens_val)

        # Compute efficiency on test set
        if self.eff_obj == "set_size":
            eff_metric_test = average_set_size(conformal_predictor, self.X_test, self.y_test, self.confidence, taxonomy=self.sens_test)

        # Compute unfairness on validaton set
        if self.fair_obj == "theoretical_cov_gap":
            fair_metric_val = 100 * np.mean(np.abs(np.array(list(cov_dict_val.values())) - (self.confidence)))
        elif self.fair_obj == "avg_priv_dif":
            fair_metric_val = 100 * avg_priv_dif(cov_dict_val, self.priviledged_value)
        elif self.fair_obj == "max_priv_dif":
            fair_metric_val = 100 * max_priv_dif(cov_dict_val, self.priviledged_value)

        # Compute unfairness on test set
        if self.fair_obj == "theoretical_cov_gap":
            fair_metric_test = 100 * np.mean(np.abs(np.array(list(cov_dict_test.values())) - (self.confidence)))
        elif self.fair_obj == "avg_priv_dif":
            fair_metric_test = 100 * avg_priv_dif(cov_dict_test, self.priviledged_value)
        elif self.fair_obj == "max_priv_dif":
            fair_metric_test = 100 * max_priv_dif(cov_dict_test, self.priviledged_value)

        cov_dict_val = {key + '_val': [value] for key, value in cov_dict_val.items()}
        cov_dict_test = {key + '_test': [value] for key, value in cov_dict_test.items()}
        eff_dict_val = {key + '_val': [value] for key, value in eff_dict_val.items()}
        eff_dict_test = {key + '_test': [value] for key, value in eff_dict_test.items()}

        cov_dict_val["marg_cov_val"] = [coverage_val]
        cov_dict_test["marg_cov_test"] = [coverage_test]

        parameters = {}
        for param in self.config[self.config["classifier"] + "_hyps"]:
            parameters[param] = learner.get_params()[param]

        individual_info = {'id': "", 'creation_mode': "micp", self.eff_obj + "_val": eff_metric_val, "unfairness_val": fair_metric_val, self.eff_obj + "_test": eff_metric_test,  "unfairness_test": fair_metric_test}
        individual_info.update(parameters)
        individual_info.update(cov_dict_val)
        individual_info.update(cov_dict_test)
        individual_info.update(eff_dict_val)
        individual_info.update(eff_dict_test)

        individuals_aux = pd.DataFrame(individual_info)
        self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            
        self.individuals_df.to_csv(
            f'./results/{self.dataset}/{self.fair_obj}/{self.classifier}/individuals/individuals_pareto_{self.sensitive_groups}{self.sensitive_attribute}_calsize_{str(self.cal_size)[2:]}_seed_{str(seed)}_gen_{str(self.generations)}_indiv_{str(self.individuals)}.csv',
            index = False,
            columns = list(individual_info.keys())
        ) 