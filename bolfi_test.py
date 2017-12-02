from pyabc.acquisition import MaxPosteriorVariance
import GPyOpt
from pyabc.prior import PriorList, Prior
import numpy as np

# --- Function to optimize
func = GPyOpt.objective_examples.experiments2d.branin()
#func.plot()


objective = GPyOpt.core.task.SingleObjective(func.f)

prior = Prior('uniform', 0, 5)

priors = PriorList([prior, prior])

model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False)

space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},
                                    {'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])

acquisition = MaxPosteriorVariance(model, space, priors)


initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)

evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design)

max_iter  = 10
bo.run_optimization(max_iter = max_iter)
