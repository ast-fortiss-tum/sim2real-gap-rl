import os
import pickle

import numpy as np
from datetime import timedelta

from commonpower.modelling import ModelHistory
from commonpower.control.controllers import RLControllerSB3, OptimalController
from commonpower.control.safety_layer.penalties import *
from commonpower.control.runners import SingleAgentTrainer, DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.logging.loggers import *
from commonpower.control.configs.algorithms import *

from commonpower.utils.param_initialization import *

import tensorboard
from models.sac import set_global_seed

class MPC:
    def __init__(self, env, save_dir="mpc_save",seed=None):
        sys = env.env.sys
        horizon=timedelta(hours=24)
        self.fixed_start="27.11.2016"

        # Set the global seed if provided for reproducibility
        if seed is not None:
            set_global_seed(seed)
        
        oc_model_history = ModelHistory([sys])
        self.oc_deployer = DeploymentRunner(
            sys=sys, 
            global_controller=OptimalController('global'), 
            forecast_horizon=horizon,
            control_horizon=horizon,
            history=oc_model_history,
            seed=seed,
            save_dir=save_dir
        )

    def train(self):
        self.oc_deployer.run(n_steps=24, fixed_start=self.fixed_start)

    # we retrieve logs for the system cost
    #oc_power_import_cost = oc_model_history.get_history_for_element(m1, name='cost') # cost for buying electricity
    #oc_dispatch_cost = oc_model_history.get_history_for_element(n1, name='cost') # cost for operating the components in the household
    #oc_total_cost = [(oc_power_import_cost[t][0], oc_power_import_cost[t][1] + oc_dispatch_cost[t][1]) for t in range(len(oc_power_import_cost))]
    #oc_soc = oc_model_history.get_history_for_element(e1, name="soc") # state of charge
