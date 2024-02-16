#!/usr/bin/env python
"""
| File: 4_python_single_vehicle.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller 
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS.
"""

# Imports to start Isaac Sim from this script
import carb
from omni.isaac.kit import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import asyncio                                                  # Used to run sample asynchronously to not block rendering thread

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with lidar sensor
from omni.isaac.core.utils.extensions import disable_extension, enable_extension

from pxr import UsdGeom, Gf, UsdPhysics, Semantics              # pxr usd imports used to create cube

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Import the custom python control backend
from utils.nonlinear_controller import NonlinearController

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

# Use os and pathlib for parsing the desired trajectory from a CSV file
import os
from pathlib import Path


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """
        enable_extension("omni.isaac.ros_bridge")

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Get the current directory used to read trajectories and save results
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        # Create the vehicle 1
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor1 = MultirotorConfig()
        config_multirotor1.backends = [NonlinearController(
            trajectory_file=self.curr_dir + "/trajectories/pitch_relay_90_deg_2.csv",
            results_file=self.curr_dir + "/results/single_statistics.npz",
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0]
        )]

        Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            [2.3, -1.5, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor1,
        )

        stage = omni.usd.get_context().get_stage()                      # Used to access Geometry
        timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
        

        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/Lidar0",
            parent="/World/quadrotor1/body",
            min_range=0.4,
            max_range=100.0,
            draw_points=True,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=60.0,
            horizontal_resolution=0.4,
            vertical_resolution=0.4,
            rotation_rate=0.0,
            high_lod=True,
            yaw_offset=0.0,
            enable_semantics=True
        )

        UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, 0.0, 0.05))

        # Create a cube, sphere, add collision and different semantic labels
        primType = ["Cube", "Sphere"]

        for i in range(2):
            prim = stage.DefinePrim("/World/"+primType[i], primType[i])
            UsdGeom.XformCommonAPI(prim).SetTranslate((-2.0, -2.0 + i * 4.0, 0.0))
            UsdGeom.XformCommonAPI(prim).SetScale((1, 1, 1))
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)

            # Add semantic label
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set("class")
            sem.GetSemanticDataAttr().Set(primType[i])

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running():

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
