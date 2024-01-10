import os
import sys
import random
import math
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import rcdefaults
rcdefaults() 
import seaborn as sns

sys.path.insert(1, "nsga2")
from nsga2.problem import Problem
from nsga2.evolution import Evolution
from nsga2.toolbox import *
from nsga2.plot import *

import warnings
warnings.filterwarnings("ignore")

# Parse experiment parameters
with open("config_file.yaml", encoding="utf-8") as yaml_file:
    config = yaml.safe_load(yaml_file)

# If experimental results folder does not exist, create it
results_path = "results/" + config["dataset"] + "/" + config["fair_obj"] + "/" + config["classifier"] 
if not os.path.exists(results_path):
    Path(results_path).mkdir(parents=True)
    Path(results_path + "/individuals").mkdir()
    Path(results_path + "/population").mkdir()

for run in range(config["n_runs"]):
    config["set_seed"] = config["set_seed"] + run
    
    problem = Problem(config = config)
    problem.create_datasets()
 
    evo = Evolution(problem, patience = 10)
 
    pareto = evo.evolve()
    problem.individuals_df = None

    first = True
    for p in pareto:    
        problem.test_and_save(p,first,problem.seed)
        first = False
 
    problem.benchmark_conformal(problem.seed)
    problem.benchmark_mondrian_conformal(problem.seed)