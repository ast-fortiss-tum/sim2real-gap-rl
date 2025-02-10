import gym
import pathlib
from datetime import timedelta, datetime

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

class SmartGridBasic(gym.Wrapper):
    def __init__(self,
                 horizon=timedelta(hours=24),
                 frequency=timedelta(minutes=60),
                 fixed_start="27.11.2016",
                 capacity=3,
                 data_path=None):
        """
        Initializes the SmartGrid environment with configurable parameters.
        
        Parameters:
            horizon (timedelta): The time horizon of the forecast/control period.
            frequency (timedelta): The time resolution for forecasting/control.
            fixed_start (str): The fixed start date as a string (format: "%d.%m.%Y").
            capacity (float): The energy storage capacity (e.g., in kWh).
            data_path (str or pathlib.Path): Optional path to the data directory. 
                If None, a default path is constructed.
        """
        # Set basic parameters.
        self.horizon = horizon
        self.frequency = frequency
        self.fixed_start = fixed_start
        self.capacity = capacity

        # Determine the data path.
        if data_path is None:
            self.current_path = pathlib.Path().absolute()
            self.data_path = (self.current_path / 'data' / '1-LV-rural2--1-sw').resolve()
        else:
            self.data_path = pathlib.Path(data_path).resolve()

        # ----------------------------
        # Setup Data Providers
        # ----------------------------
        ds1 = CSVDataSource(
            self.data_path / 'LoadProfile.csv',
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "H0-A_pload": "p", "H0-A_qload": "q"},
            auto_drop=True,
            resample=timedelta(minutes=60)
        )

        ds2 = CSVDataSource(
            self.data_path / 'LoadProfile.csv',
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
            auto_drop=True,
            resample=timedelta(minutes=60)
        )

        ds3 = CSVDataSource(
            self.data_path / 'RESProfile.csv',
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "PV3": "p"},
            auto_drop=True,
            resample=timedelta(minutes=60)
        ).apply_to_column("p", lambda x: -x)

        self.dp1 = DataProvider(ds1, LookBackForecaster(frequency=self.frequency, horizon=self.horizon))
        self.dp2 = DataProvider(ds2, LookBackForecaster(frequency=self.frequency, horizon=self.horizon))
        self.dp3 = DataProvider(ds3, PerfectKnowledgeForecaster(frequency=self.frequency, horizon=self.horizon))

        # ----------------------------
        # Setup Controller and System
        # ----------------------------
        self.agent1 = RLControllerSB3(
            name='agent1',
            safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
        )
        self.opt_controller = OptimalController("opt_ctrl")

        self.setup_system()
        self.runner_setup()

    def setup_system(self):
        # --- Define Nodes ---
        self.n1 = Bus("MultiFamilyHouse", {
            'p': (-50, 50),
            'q': (-50, 50),
            'v': (0.95, 1.05),
            'd': (-15, 15)
        })

        self.m1 = TradingBusLinear("Trading1", {
            'p': (-50, 50),
            'q': (-50, 50)
        }).add_data_provider(self.dp2)

        # --- Define Components ---
        # Create the energy storage system using the subclass's definition.
        self.e1 = self.define_e1()
        self.agent1.add_entity(self.e1)

        self.r1 = RenewableGen("PV1").add_data_provider(self.dp3)
        self.d1 = Load("Load1").add_data_provider(self.dp1)

        # --- Build the System ---
        self.sys = System(power_flow_model=PowerBalanceModel()).add_node(self.n1).add_node(self.m1)
        self.agent1.add_system(self.sys)
        self.n1.add_node(self.d1).add_node(self.e1).add_node(self.r1)

        # (Optional) Print the system structure.
        self.sys.pprint()

    def runner_setup(self):
        # --- Setup the Training Runner ---
        self.runner = SingleAgentTrainer(
            sys=self.sys,
            global_controller=self.agent1,
            wrapper=SingleAgentWrapper,
            forecast_horizon=self.horizon,
            control_horizon=self.horizon,
        )
        self.runner.fixed_start = datetime.strptime(self.fixed_start, "%d.%m.%Y")
        self.runner.prepare_run()

        # Save the created environment for later use.
        self.env = self.runner.env

    def define_e1(self):
        """
        Abstract method to define the energy storage system (ESS).
        Subclasses must override this method to specify their particular ESS implementation.
        """
        raise NotImplementedError("Subclasses must implement define_e1()")


# =============================================================================
# Subclass for a linear energy storage system.
# =============================================================================
class SmartGrid_Linear(SmartGridBasic):
    def __init__(self, **kwargs):
        """
        Initializes the SmartGridLinear environment. All parameters from the base class
        (horizon, frequency, fixed_start, capacity, data_path) can be passed as keyword arguments.
        """
        super().__init__(**kwargs)

    def define_e1(self):
        # Here, only the parameters for the ESS differ.
        return ESSLinear("ESS1", {
            'rho': 0.1,
            'p': (-1.5, 1.5),
            'q': (0, 0),
            'soc': (0.2 * self.capacity, 0.8 * self.capacity),
            "soc_init": RangeInitializer(0.2 * self.capacity, 0.8 * self.capacity)
        })


# =============================================================================
# Subclass for a nonlinear energy storage system.
# =============================================================================
class SmartGrid_Nonlinear(SmartGridBasic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def define_e1(self):
        return ESS("ESS", {
            'rho': 0.001,  # charging/discharging 1 kWh incurs a cost of wear of 0.001
            'p': (-2, 2),  # active power limits
            'q': (0, 0),  # reactive power limits
            'etac': 0.98,  # charging efficiency
            'etad': 0.98,  # discharging efficiency
            'etas': 0.99,  # self-discharge (after one time step 99% of the soc is left)
            'soc': (0.1 * self.capacity, 0.9 * self.capacity),  # soc limits
            "soc_init": ConstantInitializer(0.2 * self.capacity)  # initial soc at the start of simulation
        })

# =============================================================================
# Usage Examples:
# =============================================================================
"""
# Instantiate the linear smart grid with default parameters.
smartgrid_env_linear = SmartGrid_Linear().env

# Or, instantiate with custom parameters (e.g., a different horizon, start date, capacity, or data path):
smartgrid_env_linear_custom = SmartGrid_Linear(
    horizon=timedelta(hours=12),
    frequency=timedelta(minutes=30),
    fixed_start="01.01.2020",
    capacity=5,
    data_path="/path/to/your/data"
).env
"""

