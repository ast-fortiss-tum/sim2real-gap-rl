from __future__ import absolute_import, division, print_function
import pathlib
from datetime import timedelta, datetime
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

import pathlib
import matplotlib.pyplot as plt
from functools import partial
from commonpower.modelling import ModelHistory
from commonpower.core import System, Node, Bus
from commonpower.models.busses import *
from commonpower.models.components import *
from commonpower.models.powerflow import *
from commonpower.control.controllers import RLControllerSB3, OptimalController, RLBaseController
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.safety_layer.penalties import *
from commonpower.control.runners import BaseTrainer, BaseRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.logging.loggers import *
from commonpower.control.configs.algorithms import *
from commonpower.data_forecasting import *
from commonpower.utils.helpers import get_adjusted_cost
from commonpower.utils.param_initialization import *
from commonpower.control.environments import ControlEnv   

class SmartGridBasic:
    def __init__(self,
                 rl = True,
                 policy_path = None,
                 horizon=timedelta(hours=24),
                 frequency=timedelta(minutes=60),
                 fixed_start="27.11.2016",
                 capacity=3,
                 data_path=None,
                 params_battery=None):
        """
        Initializes the SmartGrid environment with configurable parameters.
        
        Parameters:
            horizon (timedelta): The time horizon of the forecast/control period.
            frequency (timedelta): The time resolution for forecasting/control.
            fixed_start (str): The fixed start date as a string (format: "%d.%m.%Y").
            capacity (float): The energy storage capacity (e.g., in kWh).
            data_path (str or pathlib.Path): Optional path to the data directory. 
                If None, a default path is constructed.
            params_battery (dict): A dictionary of battery parameters.
        """
        # Set basic parameters.
        self.rl = rl

        self.horizon = horizon
        self.frequency = frequency
        self.fixed_start = fixed_start
        self.capacity = capacity
        self.params_battery = params_battery

        # Determine the data path.
        if data_path is None:
            self.current_path = pathlib.Path().absolute()
            self.data_path = (self.current_path / 'data' / '1-LV-rural2--1-sw').resolve()
        else:
            self.data_path = pathlib.Path(data_path).resolve()

        # ----------------------------
        # Setup Data Providers
        # ----------------------------
        d11 = CSVDataSource(
            self.data_path / 'LoadProfile.csv',
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
            auto_drop=True,
            resample=timedelta(minutes=60)
        )

        ds1 = ConstantDataSource({
            "psis": 0.08,  # the household gets payed 0.08 price units for each kWh exported to the external grid
            "psib": 0.34  # the houshold pays 0.34 price units for each imported kWh
            }, 
            date_range=d11.get_date_range(), 
            frequency=timedelta(minutes=60)
        )

        ds2 = CSVDataSource(
            self.data_path / 'LoadProfile.csv',
            delimiter=";",
            datetime_format="%d.%m.%Y %H:%M",
            rename_dict={"time": "t", "H0-A_pload": "p", "H0-A_qload": "q"},
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
            pretrained_policy_path=policy_path
        )
        self.opt_controller = OptimalController("opt_ctrl")

        # Call setup methods.
        self.setup_system()
        self.setup_runner_trainer(rl = self.rl)

    def setup_system(self):
        # --- Define Nodes ---
        """self.n1 = Bus("MultiFamilyHouse", {
            'p': (-50, 50),
            'q': (-50, 50),
            'v': (0.95, 1.05),
            'd': (-15, 15)
        })"""

        self.n1 = RTPricedBus("Household").add_data_provider(self.dp1)
        self.m1 = ExternalGrid("ExternalGrid")
        """
        self.m1 = TradingBusLinear("Trading1", {
            'p': (-50, 50),
            'q': (-50, 50)
        }).add_data_provider(self.dp2)
        
        """

        # --- Define Components ---
        # Create the energy storage system using the subclass's definition.
        self.e1 = self.define_e1()

        self.r1 = RenewableGen("PV1").add_data_provider(self.dp3)
        self.d1 = Load("Load1").add_data_provider(self.dp2)

        # --- Build the System ---
        self.sys = System(power_flow_model=PowerBalanceModel()).add_node(self.n1).add_node(self.m1)
        self.n1.add_node(self.d1).add_node(self.e1).add_node(self.r1)

        # (Optional) Print the system structure.
        self.sys.pprint()

    def setup_runner_trainer(self, rl = True):
        # --- Setup the Trainer ---
        if rl:
            self.runner = None
            self.trainer = BaseTrainer(
                sys=self.sys,
                global_controller=self.agent1,
                wrapper=SingleAgentWrapper,
                forecast_horizon=self.horizon,
                control_horizon=self.horizon,
            )
            self.trainer.fixed_start = datetime.strptime(self.fixed_start, "%d.%m.%Y")
            self.trainer.prepare_run()
            self.env = self.trainer.env

        else:
            self.trainer = False
            self.runner = BaseRunner(
                sys=self.sys,
                global_controller=self.opt_controller,
                forecast_horizon=self.horizon,
                control_horizon=self.horizon,
            )
            self.runner.fixed_start = datetime.strptime(self.fixed_start, "%d.%m.%Y")
            self.runner.prepare_run()
            self.env_opt = self.runner.env

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
        # Ensure battery parameters are provided.
        if "params_battery" not in kwargs or kwargs["params_battery"] is None:
            raise ValueError("The 'params_battery' parameter must be provided for SmartGrid_Linear.")
        # Set battery-specific attributes before calling the base class constructor.
        self.params_battery = kwargs["params_battery"]
        self.rho = self.params_battery["rho"]      # wear cost per kWh
        self.p_lim = self.params_battery["p_lim"]    # power limits
        
        # Now call the base class constructor.
        super().__init__(**kwargs)

    def define_e1(self):
        # Define the energy storage system (ESS) for the linear case.
        return ESSLinear("ESS1", {
            'rho': self.rho,
            'p': (-self.p_lim, self.p_lim),
            'q': (0, 0),
            'soc': (0.1 * self.capacity, 0.9 * self.capacity),
            "soc_init": ConstantInitializer(0.2 * self.capacity)
        })

# =============================================================================
# Subclass for a nonlinear energy storage system.
# =============================================================================
class SmartGrid_Nonlinear(SmartGridBasic):
    def __init__(self, **kwargs):
        # Ensure battery parameters are provided.
        if "params_battery" not in kwargs or kwargs["params_battery"] is None:
            raise ValueError("The 'params_battery' parameter must be provided for SmartGrid_Nonlinear.")
        # Set battery-specific attributes before calling the base class constructor.
        self.params_battery = kwargs["params_battery"]
        self.rho = self.params_battery["rho"]      # wear cost per kWh
        self.p_lim = self.params_battery["p_lim"]    # power limits
        self.etac = self.params_battery["etac"]      # charging efficiency
        self.etad = self.params_battery["etad"]      # discharging efficiency
        self.etas = self.params_battery["etas"]      # self-discharge
        super().__init__(**kwargs)

    def define_e1(self):
        # Define the energy storage system (ESS) for the nonlinear case.
        return ESS("ESS", {
            'rho': self.rho,
            'p': (-self.p_lim, self.p_lim),
            'q': (0, 0),
            'etac': self.etac,
            'etad': self.etad,
            'etas': self.etas,
            'soc': (0.1 * self.capacity, 0.9 * self.capacity),
            "soc_init": ConstantInitializer(0.2 * self.capacity)
        })


# =============================================================================
# Usage Examples:
# =============================================================================
"""
# Define battery parameters in a dictionary.
battery_params = {
    "rho": 0.1,
    "p_lim": 1.5,
    "etac": 0.95,
    "etad": 0.95,
    "etas": 0.99
}

# Instantiate the linear smart grid with default parameters.
smartgrid_env_linear = SmartGrid_Linear(params_battery=battery_params).env

# Or, instantiate with custom parameters:
smartgrid_env_linear_custom = SmartGrid_Linear(
    horizon=timedelta(hours=12),
    frequency=timedelta(minutes=30),
    fixed_start="01.01.2020",
    capacity=5,
    data_path="/path/to/your/data",
    params_battery=battery_params
).env
"""
