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
from commonpower.control.controllers import RLControllerSB3, OptimalController, RLBaseController, RLControllerSAC_Customized
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.safety_layer.penalties import *
from commonpower.control.runners import BaseTrainer, BaseRunner, DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.logging.loggers import *
from commonpower.control.configs.algorithms import *
from commonpower.data_forecasting import *
from commonpower.utils.helpers import get_adjusted_cost
from commonpower.utils.param_initialization import *
from commonpower.control.environments import ControlEnv   

class SmartGridBasic:
    def __init__(self,
                 rl=True,
                 policy_path=None,
                 horizon=timedelta(hours=24),
                 frequency=timedelta(minutes=60),
                 fixed_start=None,
                 capacity=3,
                 data_path=None,
                 seed=42,
                 params_battery=None):
        """
        Initializes the SmartGrid environment with configurable parameters.
        Only basic instance variables and objects are defined here.
        To complete the setup, explicitly call setup_system() and setup_runner_trainer() as needed.
        """
        self.seed = seed
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
            "psis": 0.08,  # price units for kWh exported to the external grid
            "psib": 0.34   # price units for kWh imported from the external grid
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
        self.agent1 = RLControllerSAC_Customized(
            name='agent1',
            safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
            pretrained_policy_path=policy_path
        )
        self.opt_controller = OptimalController("opt_ctrl")

    def setup_system(self):
        # --- Define Nodes and Components (for a single household, not used in two-house grid) ---
        self.n1 = RTPricedBus("Household").add_data_provider(self.dp1)
        self.m1 = ExternalGrid("ExternalGrid")
        self.e1 = self.define_e1()
        self.r1 = RenewableGen("PV1").add_data_provider(self.dp3)
        self.d1 = Load("Load1").add_data_provider(self.dp2)
        self.sys = System(power_flow_model=PowerBalanceModel()).add_node(self.n1).add_node(self.m1)
        self.n1.add_node(self.d1).add_node(self.e1).add_node(self.r1)
        self.sys.pprint()

    def setup_runner_trainer(self, rl=True):
        # --- Setup the Trainer or Runner ---
        #self.sys.controllers = []
        print("Controllers in system:", self.sys.controllers)
        if rl:
            self.runner = None
            self.trainer = BaseTrainer(
                seed=self.seed,
                sys=self.sys,
                global_controller=self.agent1,
                wrapper=SingleAgentWrapper,
                forecast_horizon=self.horizon,
                control_horizon=self.horizon,
            )
            if self.fixed_start is not None:
                self.trainer.fixed_start = datetime.strptime(self.fixed_start, "%d.%m.%Y")
            else:
                self.trainer.fixed_start = None
            self.trainer.prepare_run()
            self.env = self.trainer.env
        else:
            self.trainer = False
            self.runner = DeploymentRunner(
                seed=self.seed,
                sys=self.sys,
                global_controller=self.opt_controller,
                forecast_horizon=self.horizon,
                control_horizon=self.horizon,
            )
            if self.fixed_start is not None:
                self.runner.fixed_start = datetime.strptime(self.fixed_start, "%d.%m.%Y")
            else:
                self.runner.fixed_start = None
            self.runner.prepare_run()
            self.env = self.runner.env

    def define_e1(self):
        raise NotImplementedError("Subclasses must implement define_e1()")

# =============================================================================
# Subclass for a linear energy storage system.
# =============================================================================
class SmartGrid_Linear(SmartGridBasic):
    def __init__(self, **kwargs):
        if "params_battery" not in kwargs or kwargs["params_battery"] is None:
            raise ValueError("The 'params_battery' parameter must be provided for SmartGrid_Linear.")
        self.params_battery = kwargs["params_battery"]
        self.rho = self.params_battery["rho"]      # wear cost per kWh
        self.p_lim = self.params_battery["p_lim"]    # power limits
        super().__init__(**kwargs)

    def define_e1(self):
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
        if "params_battery" not in kwargs or kwargs["params_battery"] is None:
            raise ValueError("The 'params_battery' parameter must be provided for SmartGrid_Nonlinear.")
        self.params_battery = kwargs["params_battery"]
        self.rho = self.params_battery["rho"]
        self.p_lim = self.params_battery["p_lim"]
        self.etac = self.params_battery["etac"]
        self.etad = self.params_battery["etad"]
        self.etas = self.params_battery["etas"]
        super().__init__(**kwargs)

    def define_e1(self):
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
# Subclass for a two-house smart grid.
# =============================================================================
class SmartGrid_TwoHouses(SmartGridBasic):
    def __init__(self, battery2_damaged=False, **kwargs):
        """
        Initializes the two-house smart grid environment.
        Only sets up instance variables. To build the full system,
        call setup_system() explicitly after initialization.
        """
        if "params_battery" not in kwargs or kwargs["params_battery"] is None:
            raise ValueError("The 'params_battery' parameter must be provided for SmartGrid_TwoHouses.")
        self.params_battery = kwargs["params_battery"]
        self.rho = self.params_battery["rho"]
        self.p_lim = self.params_battery["p_lim"]
        self.capacity = kwargs.get("capacity", 3)
        self.battery2_damaged = battery2_damaged
        super().__init__(**kwargs)

    def setup_system(self):
        """
        Setup a system with two households (each with a battery, PV, and load)
        connected to an ExternalGrid. Battery 2 is configured as damaged if indicated.
        """
        self.n1 = RTPricedBus("Household1").add_data_provider(self.dp1)
        self.n2 = RTPricedBus("Household2").add_data_provider(self.dp1)
        self.m1 = ExternalGrid("ExternalGrid")
        
        battery1 = self.define_battery(1)
        pv1 = RenewableGen("PV_house1").add_data_provider(self.dp3)
        load1 = Load("Load_house1").add_data_provider(self.dp2)
        self.n1.add_node(battery1).add_node(pv1).add_node(load1)
        
        battery2 = self.define_battery(2)
        pv2 = RenewableGen("PV_house2").add_data_provider(self.dp3)
        load2 = Load("Load_house2").add_data_provider(self.dp2)
        self.n2.add_node(battery2).add_node(pv2).add_node(load2)
        
        self.sys = System(power_flow_model=PowerBalanceModel()) \
                    .add_node(self.n1) \
                    .add_node(self.n2) \
                    .add_node(self.m1)
                    
        self.sys.pprint()

    def define_battery(self, house_index):
        """
        Helper method to define the battery for a given house.
        If house_index is 2 and battery2_damaged is True, reduced efficiencies are used.
        """
        if house_index == 2 and self.battery2_damaged:
            etac_used = 0.1
            etad_used = 0.1
            etas_used = 0.1
            name = f"ESS_house{house_index}_damaged"
        else:
            etac_used = self.params_battery["etac"]
            etad_used = self.params_battery["etad"]
            etas_used = self.params_battery["etas"]
            name = f"ESS_house{house_index}"
        
        return ESS(name, {
            'rho': self.rho,
            'p': (-self.p_lim, self.p_lim),
            'q': (0, 0),
            'etac': etac_used,
            'etad': etad_used,
            'etas': etas_used,
            'soc': (0.1 * self.capacity, 0.9 * self.capacity),
            "soc_init": ConstantInitializer(0.2 * self.capacity)
        })

    # We still need to define define_e1 even though it is not used in this subclass.
    def define_e1(self):
        raise NotImplementedError("SmartGrid_TwoHouses uses define_battery() for battery definition.")
