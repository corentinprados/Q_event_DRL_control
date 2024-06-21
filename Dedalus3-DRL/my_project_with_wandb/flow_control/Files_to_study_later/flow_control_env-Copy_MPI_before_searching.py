# flow_control/flow_control_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import dedalus.public as de
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from dedalus.extras import plot_tools
import matplotlib.pyplot as plt
import os
import pickle
from mpi4py import MPI

#####################
#### ENVIRONMENT ####
#####################

class FlowControlEnv(gym.Env):
    def __init__(self):
        super(FlowControlEnv, self).__init__()        
        
        # Initialize MPI components
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MPI initialized with rank {self.rank} and size {self.size}")

        
        # Define the grid dimensions and dealias factor
        self.Nx, self.Nz = 64, 16
        self.dealias = 3/2
        Nx_dealias = int(self.Nx * self.dealias)
        Nz_dealias = int(self.Nz * self.dealias)
        
        # Define action space: a single continuous parameter alpha between -alpha_max and alpha_max
        self.alpha_max = 0.03  # Maximum value for alpha
        #### Why 0.3 ? Because with a fcator of 1, in 0.8s the transition function is at 0.3
        #### Why 0.1 because the change where to big 
        #### Why 0.05 because the change seems still too big 
        self.action_space = spaces.Box(low=-self.alpha_max, high=self.alpha_max, shape=(1,), dtype=np.float32)
        
        # Define observation space based on actual grid dimensions and number of components
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(Nx_dealias, Nz_dealias, 2), dtype=np.float32)
        
        # Initialize Dedalus simulation parameters
        self.solver, self.flow, self.CFL, self.u, self.ex, self.ez, self.dist, self.coords = self.init_dedalus_simulation()
        
        # Initialize additional parameters for RL agent actions
        self.factor = 0.3       # Growth factor for exponential
        #### Why 0.5 because the change where to quick 
        self.t0 = 1           # Time interval between agent activations
        self.t_i = self.t0    # Initial activation time
        self.agent_active = False
        self.alpha_final = 0
        self.signe = 1
        self.max_simulation_time = 70
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize list to store rewards 
        self.rewards = []

    def init_dedalus_simulation(self):
        """
        Initializes the Dedalus simulation including setting up the domain, problem, solver, CFL condition, logging, checkpoints, and snapshots.
        """
        # Setup logger
        logger = logging.getLogger(__name__)
        
        # Simulation parameters
        Lx, Lz = 4, 1
        self.Nx, self.Nz = 64, 16
        Re = 1e4
        self.dealias = 3/2
        
        # Stopping criteria and restart settings
        self.initial_time = 950
        RL_time = 50
        stop_sim_time = self.initial_time + RL_time
        checkpoints_sim_dt = 50
        snap_sim_dt = 1
        
        # Time stepper and time step settings
        timestepper = de.RK222
        dt0 = 1e-8
        max_timestep = 0.125
        dtype = np.float64
        
        # Set up the bases for the simulation domain
        coords = de.CartesianCoordinates('x', 'z')
        dist = de.Distributor(coords, dtype=dtype)
        
        xbasis = de.RealFourier(coords['x'], size=self.Nx, bounds=(0, Lx), dealias=self.dealias)
        zbasis = de.ChebyshevT(coords['z'], size=self.Nz, bounds=(-Lz, Lz), dealias=self.dealias)

        # Second base for running to take into account the delias, call it the ruuning base 
        x2basis = de.RealFourier(coords['x'], size=int(64 * 3 / 2), bounds=(0, 4), dealias=self.dealias)
        z2basis = de.ChebyshevT(coords['z'], size=int(16 * 3 / 2), bounds=(-1, 1), dealias=self.dealias)
        x2, z2 = dist.local_grids(x2basis, z2basis)
        self.x2 = x2  # Store x2 for later use
        self.z2 = z2  # Store z2 for later use
        
        # Define fields for pressure and velocity
        p = dist.Field(name='p', bases=(xbasis, zbasis))
        u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
        
        # Get local grids
        x, z = dist.local_grids(xbasis, zbasis)
        self.z = z  # Store z for later use
        
        zfield = dist.Field(name='zfield', bases=(xbasis, zbasis))
        zfield['g'] = z
        
        # Define unit vectors
        ex, ez = coords.unit_vector_fields(dist)
        
        # Tau method for boundary conditions
        tau_p = dist.Field(name='tau_p')
        tau_ux = dist.VectorField(coords, name='tau_ux', bases=xbasis)
        tau_uz = dist.VectorField(coords, name='tau_uz', bases=xbasis)
        
        lift_basis = zbasis.derivative_basis(1)
        lift = lambda A: de.Lift(A, lift_basis, -1)
        grad_u = de.grad(u) + ez * lift(tau_ux)
        
        # Define the problem (initial value problem)
        problem = de.IVP([p, u, tau_p, tau_ux, tau_uz], namespace=locals())
        problem.add_equation("trace(grad_u) + tau_p = 0")  # Continuity equation
        problem.add_equation("dt(u) - 1/Re*div(grad_u) + grad(p) + lift(tau_uz) = - u @ grad(u) + 2/Re * ex")  # Navier-Stokes equation
        problem.add_equation("u(z=1) = 0")  # Boundary condition at the top
        problem.add_equation("u(z=-1) = 0")  # Boundary condition at the bottom
        problem.add_equation("integ(p) = 0")  # Pressure gauge condition
        
        # Build solver
        solver = problem.build_solver(timestepper)
        solver.stop_sim_time = stop_sim_time
        
        # CFL condition for adaptive time stepping
        CFL = de.CFL(solver, initial_dt=dt0, cadence=10, safety=0.1, threshold=0.05,
                     max_change=1.5, min_change=0.5, max_dt=max_timestep)
        CFL.add_velocity(u)
        
        # Initial conditions and checkpoint handling
        write, initial_timestep = solver.load_state(f'checkpoints_initial_conditions/checkpoints_s24.h5')
        initial_timestep = dt0
        file_handler_mode = 'append'
        logger.info("Resuming from checkpoint")

        # Flow property diagnostics
        flow = de.GlobalFlowProperty(solver, cadence=10)
        # For now, the flow is directly watch through .render()
        """
        flow.add_property(np.sqrt(u @ u) * Re, name='Re')
        
        # Setup checkpoint file handler
        checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=checkpoints_sim_dt, max_writes=1, mode=file_handler_mode)
        checkpoints.add_tasks(solver.state)
        
        # Setup snapshot file handler
        snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snap_sim_dt, max_writes=50, mode=file_handler_mode)
        snapshots.add_task(de.dot(ex, u), name='U_x')
        snapshots.add_task(de.dot(ez, u), name='U_y')
        """
        
        # Calculate delta_N0 for initial state
        self.delta_N0 = self.compute_delta_N(u['g'][0], u['g'][1], self.z2)
        
        return solver, flow, CFL, u, ex, ez, dist, coords

    def gather_velocity_fields(self):
        U_local = self.u['g'][0]
        V_local = self.u['g'][1]
    
        # Gather U and V from all ranks
        U = np.zeros((U_local.shape[0], U_local.shape[1] * self.size))
        V = np.zeros((V_local.shape[0], V_local.shape[1] * self.size))
        self.comm.Gather(U_local, U, root=0)
        self.comm.Gather(V_local, V, root=0)
    
        if self.rank == 0:
            return U, V
        else:
            return None, None

    
    def compute_delta_N(self, U, V, z):
        """
        Computes the deviation Delta_N from the base state.
        """
        U_base = 1 - z**2
        V_base = np.zeros_like(z)
        delta_N = np.sqrt(np.sum((U - U_base)**2 + (V - V_base)**2))
        return delta_N

    def compute_reward(self, observation):
        """
        Computes the reward based on the deviation from the base state.
        """
        U, V = self.gather_velocity_fields()
        if self.rank == 0:
            delta_N = self.compute_delta_N(U, V, self.z2)
            reward = 1 - (delta_N / self.delta_N0)
            return reward
        else:
            return 0

    def get_observation(self):
        """
        Extracts the velocity fields as the observation.
        """
        U, V = self.gather_velocity_fields()
        if self.rank == 0:
            observation = np.stack([U, V], axis=-1)
        else:
            observation = None
        return observation
    
    def render(self, mode='human'):
        """
        Visualizes the flow field using plt.pcolormesh.
        """
        U, V = self.gather_velocity_fields()
        if self.rank == 0:
            # Clear previous plot
            plt.clf()

            # Create a figure
            fig, axes = plt.subplots(2, 1, figsize=(8, 8))

            # Get the 2D meshes for x and z from the Dedalus domain
            x_mesh, z_mesh = np.meshgrid(self.x2, self.z2)

            # Plot U_x component
            im1 = axes[0].pcolormesh(x_mesh, z_mesh, U.T, cmap='coolwarm')
            fig.colorbar(im1, ax=axes[0])
            axes[0].set_title('U_x')
            axes[0].set_aspect('equal')

            # Plot U_z component
            im2 = axes[1].pcolormesh(x_mesh, z_mesh, V.T, cmap='coolwarm')
            fig.colorbar(im2, ax=axes[1])
            axes[1].set_title('U_z')
            axes[1].set_aspect('equal')

    
    def check_done(self, observation):
        """
        Checks if the simulation has reached the end time, considering the initial simulation time.
        """
        if self.rank == 0:
            max_simulation_time = self.max_simulation_time
            if self.solver.sim_time >= self.initial_time + max_simulation_time:
                self.logger.warning('Maximum simulation time reached')
                return True

            # Check for divergence
            U, V = self.gather_velocity_fields()
            max_velocity = np.max(np.sqrt(U**2 + V**2))
            divergence_threshold = 1e3

            if max_velocity > divergence_threshold:
                self.logger.warning('Simulation diverged with max velocity: %f', max_velocity)
                return True

            return False
        else:
            return False

    def reset(self, seed=None, **kwargs):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        # Seed the environment for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        self.solver, self.flow, self.CFL, self.u, self.ex, self.ez, self.dist, self.coords = self.init_dedalus_simulation()
        self.t_i = self.t0
        self.agent_active = False
        initial_observation = self.get_observation()
        return initial_observation, {}

    
    def prepare_agent_action(self, action):
        """
        Prepares the agent action by setting the final alpha value based on the action taken by the agent.
        """
        self.alpha_final = action[0]
        self.signe = np.sign(self.alpha_final)
        # self.logger.info('alpha_final=%f', self.alpha_final) # Add this line if you want to see the value of alpha the aggent chose
        self.t_i += self.t0
        self.agent_active = True
    
    def apply_gradual_jets(self, t):
        U, V = self.gather_velocity_fields()
        if self.rank == 0:
            alpha = np.heaviside(t - (self.t_i - self.t0), 1) * np.exp(-1 / (self.factor * (t - (self.t_i - self.t0))))
            if np.abs(self.alpha_final) - alpha > 0:
                N = 4
                alphas = [0, 0, self.signe * alpha, 0]
                jets = self.jets(N, alphas, self.x2, self.z2)
                U += jets  # Apply jets to the flow
                V += jets  # Apply jets to the flow
            else:
                self.agent_active = False
    
            # Scatter the updated fields back to the ranks
            self.comm.Scatter(U, self.u['g'][0], root=0)
            self.comm.Scatter(V, self.u['g'][1], root=0)


    def jets(self, N, alphas, x, z):
        """
        Generate an initial condition in the flow, creating N jets at the center (z = 0) with specified strengths (alphas).
        """
        jets = 0
        epsilon = 0.5
        z_heaviside = np.heaviside(z + epsilon, 1) - np.heaviside(z, 1)
        x_position = np.linspace(0, 4, N + 1)
        for i in range(N):
            alpha = alphas[i]
            jets += alpha * (np.heaviside(x - x_position[i], 1) - np.heaviside(x - x_position[i + 1], 1))
        return jets * z_heaviside

    def step(self, action):
        """
        Advances the simulation by one timestep. Checks if it's time for the agent to act, prepares the agent action, and gradually applies the jets.
        """
        timestep = self.CFL.compute_timestep()
        self.solver.step(timestep)

        t = self.solver.sim_time - self.initial_time

        if self.solver.iteration % 1000 == 0:
            self.render()
            self.logger.info('Time=%e' % (t))
            plt.pause(0.5)
            plt.close()

        # Check if it's time to activate the agent
        if t - self.t_i > 0:
            self.prepare_agent_action(action)

        # Apply the jets gradually
        if self.agent_active:
            self.apply_gradual_jets(t)

        observation = self.get_observation()
        reward = self.compute_reward(observation)
        done = self.check_done(observation)

        # Store reward and observation
        self.rewards.append(reward)

        return observation, reward, done, {}, {}
