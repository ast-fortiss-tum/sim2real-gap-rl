from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym
#import gymnasium as gym

import pathlib
#import wandb
import matplotlib.pyplot as plt
from functools import partial
from commonpower.modelling import ModelHistory
from commonpower.core import System, Node, Bus
from commonpower.models.busses import *
from commonpower.models.components import *
from commonpower.models.powerflow import *
from commonpower.control.controllers import RLControllerSB3, OptimalController
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.safety_layer.penalties import *
from commonpower.control.runners import SingleAgentTrainer, DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.logging.loggers import *
from commonpower.control.configs.algorithms import *
from commonpower.data_forecasting import *
from commonpower.utils.helpers import get_adjusted_cost
from commonpower.utils.param_initialization import *
from commonpower.control.environments import ControlEnv   
from commonpower.control.controllers import RLBaseController
#from stable_baselines3 import *     
from datetime import datetime

class SmartGrid_basic(gym.Wrapper):
    def __init__(self):
        # Data providers
        self.horizon = timedelta(hours=24)
        self.frequency = timedelta(minutes=60)
        self.fixed_start = "27.11.2016"

        # path to data profiles
        self.current_path = pathlib.Path().absolute()
        self.data_path = self.current_path / 'data' / '1-LV-rural2--1-sw'
        self.data_path = self.data_path.resolve()

        ds1 = CSVDataSource(self.data_path  / 'LoadProfile.csv',
                    delimiter=";", 
                    datetime_format="%d.%m.%Y %H:%M", 
                    rename_dict={"time": "t", "H0-A_pload": "p", "H0-A_qload": "q"},
                    auto_drop=True, 
                    resample=timedelta(minutes=60))

        ds2 = CSVDataSource(self.data_path / 'LoadProfile.csv',
                    delimiter=";", 
                    datetime_format="%d.%m.%Y %H:%M", 
                    rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
                    auto_drop=True, 
                    resample=timedelta(minutes=60))

        ds3 = CSVDataSource(self.data_path / 'RESProfile.csv', 
                delimiter=";", 
                datetime_format="%d.%m.%Y %H:%M", 
                rename_dict={"time": "t", "PV3": "p"},
                auto_drop=True, 
                resample=timedelta(minutes=60)).apply_to_column("p", lambda x: -x)

        self.dp1 = DataProvider(ds1, LookBackForecaster(frequency=self.frequency, horizon=self.horizon))
        self.dp2 = DataProvider(ds2, LookBackForecaster(frequency=self.frequency, horizon=self.horizon))
        self.dp3 = DataProvider(ds3, PerfectKnowledgeForecaster(frequency=self.frequency, horizon=self.horizon))

        # System 1 definition
        self.agent1 = RLControllerSB3(
            name='agent1', 
            safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
        )   

        self.opt_controller = OptimalController("opt_ctrl")

class SmartGrid_linear(SmartGrid_basic):
    def __init__(self):
        super().__init__(self)

        # nodes
        self.n1 = Bus("MultiFamilyHouse", {
            'p': (-50, 50),
            'q': (-50, 50),
            'v': (0.95, 1.05),
            'd': (-15, 15)
        })

        # trading unit with price data for buying and selling electricity (to reduce problem complexity, we assume that
        # prices for selling and buying are the same --> TradingLinear)
        self.m1 = TradingBusLinear("Trading1", {
            'p': (-50, 50),
            'q': (-50, 50)
        }).add_data_provider(dp2)

        #opt_controller.add_entity(m1)

        # components
        # energy storage sytem
        self.capacity = 3  #kWh
        self.e1 = ESSLinear("ESS1", {
            'rho': 0.1, 
            'p': (-1.5, 1.5), 
            'q': (0, 0), 
            'soc': (0.2 * self.capacity, 0.8 * self.capacity), 
            "soc_init": RangeInitializer(0.2 * self.capacity, 0.8 * self.capacity)
        })  

        self.agent1.add_entity(self.e1)

        # photovoltaic with generation data
        self.r1 = RenewableGen("PV1").add_data_provider(self.dp3)

        # static load with data source
        self.d1 = Load("Load1").add_data_provider(self.dp1)

        # we first have to add the nodes to the system 
        # and then add components to the node in order to obtain a tree-like structure
        self.sys = System(power_flow_model=PowerBalanceModel()).add_node(self.n1).add_node(self.m1)
        #opt_controller.add_system(sys)
        self.agent1.add_system(self.sys)

        # add components to nodes
        self.n1.add_node(self.d1).add_node(self.e1).add_node(self.r1)

        # show system structure: 
        self.sys.pprint()

        runner = SingleAgentTrainer(
            sys=self.sys, 
            global_controller=self.agent1, 
            wrapper=SingleAgentWrapper, 
            #alg_config=self.alg_config, 
            forecast_horizon=self.horizon,
            control_horizon=self.horizon,
            #logger=self.logger,
            #save_path=model_path_1, 
            #seed=alg_config.seed
        )

        runner.fixed_start = datetime.strptime(self.fixed_start, "%d.%m.%Y")
        runner.prepare_run()
        SmartGrid_env = runner.env

        return SmartGrid_env

class SmartGrid_cuadratic(SmartGrid_basic):
    def __init__(self):
        super().__init__(self)


        # nodes
        n2 = Bus("MultiFamilyHouse", {
            'p': (-50, 50),
            'q': (-50, 50),
            'v': (0.95, 1.05),
            'd': (-15, 15)
        })

        # trading unit with price data for buying and selling electricity (to reduce problem complexity, we assume that
        # prices for selling and buying are the same --> TradingLinear)
        m2 = TradingBusLinear("Trading1", {
            'p': (-50, 50),
            'q': (-50, 50)
        }).add_data_provider(dp2)

        #opt_controller.add_entity(m1)

        # components
        # energy storage sytem
        capacity = 3  #kWh
        e2 = ESS("ESS1", {
            'rho': 0.1, 
            'p': (-1.5, 1.5), 
            'q': (0, 0), 
            'etas': 0.9,
            'etac': 0.9,
            'etad': 0.9,
            'soc': (0.2 * capacity, 0.8 * capacity), 
            "soc_init": RangeInitializer(0.2 * capacity, 0.8 * capacity)
        })  

        agent2.add_entity(e2)

        # photovoltaic with generation data
        r2 = RenewableGen("PV1").add_data_provider(dp3)

        # static load with data source
        d2 = Load("Load1").add_data_provider(dp1)

        # we first have to add the nodes to the system 
        # and then add components to the node in order to obtain a tree-like structure
        sys2 = System(power_flow_model=PowerBalanceModel()).add_node(n2).add_node(m2)
        #opt_controller.add_system(sys)
        agent2.add_system(sys2)

        # add components to nodes
        n2.add_node(d2).add_node(e2).add_node(r2)

        # show system structure: 
        sys2.pprint()

        return SingleAgentWrapper(env)
