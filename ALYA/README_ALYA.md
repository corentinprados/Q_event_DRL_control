# ALYA Repository Overview

This document provides an overview of the contents within the ALYA repository, which includes all necessary files and folders for conducting ALYA simulations.

## Folder Structure

Below is a description of the main folders within the repository:

| Folder Name                         | Description |
|-------------------------------------|-------------|
| `re180_min_channel_initial_code`    | Contains the initial setup for running simulations at a Reynolds number of 180. Everything is pre-configured for immediate simulation. |
| `re180_min_channel_1`               | This folder holds the output from a simulation run with a simulation time of 1. See the table below for details on key files and their configurations. |
| `longer_run_vtk`                    | Contains VTK files converted from specific ALYA output files (`mpio.run`) using the command: `mpirun -np 6 /scratch/polsm/alya_exe/mpio2vtk channel`. |

### Detailed Configuration of `re180_min_channel_1`

| File Name          | Parameter           | Description |
|--------------------|---------------------|-------------|
| `channel.dat`      | Timestep            | Set to 1. Use `mpirun -np 6 ./alya_3D.x channel` for execution, add `nohup` prefix for background processing. |
| `channel.ker.dat`  | Steps per output    | Configured for 50 steps before each output; tailored to maintain a Reynolds number of 180. |
| `channel.nsi.dat`  | Postprocess setting | Set to `VELOC` to save all velocity fields, allowing for detailed post-simulation analysis. |

## Key Files

| File Name                  | Description |
|----------------------------|-------------|
| `Re180.std`                | Data from Jimenez's paper used for comparison and validation of our simulations. |
| `ALYA_post_processing.ipynb` | Jupyter Notebook for analyzing ALYA simulation data. |

## Jupyter Notebook Details

The Jupyter Notebook included in this repository (`ALYA_post_processing.ipynb`) is structured as follows:

### 0. Introduction
- **Libraries**: Importation of libraries necessary for processing the data.

### 1. Data Preparation
- **Loading Data**: Ingest the data required for analysis.
- **Normalization**: Standardize the data to a common scale.
- **Velocity Fields Definitions**: Define the various velocity fields used in simulations.

### 2. Validation of the Simulation
- **Data Loading**: Load simulation outputs.
- **Rewriting of ALYA Data**: Transform data into a format suitable for analysis.
- **Mean Streamwise Velocity Field**: Analyze the mean velocity field in the streamwise direction.
- **Root-Mean-Square Velocity Fluctuations**: Evaluate the fluctuations in velocity.
- **Reynolds Shear Stress**: Examine the shear stress due to Reynolds number.
- **Conclusion**: Summarize the findings from the validation.

### 3. Q-Events Detection
- **Explanation**: Describe the theory and rationale behind Q-events detection.
- **In Practice**: Steps for detecting, counting, and analyzing Q-events using percolation diagrams.

This README aims to clarify the structure and content of the ALYA repository for users, ensuring effective navigation and utilization of the provided simulation tools and data.