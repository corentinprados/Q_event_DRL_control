# ALYA

## Description
This folder contains all files related to ALYA simulations for the project "Q_event_DRL_control". The ALYA simulations are crucial for understanding and controlling Q events in fluid dynamics using Deep Reinforcement Learning.

## Folder Structure
The repository is organized into the following main folders:

| Folder Name                         | Description |
|-------------------------------------|-------------|
| `re180_min_channel_initial_code`    | Contains the initial setup for running simulations at a Reynolds number of 180. Everything is pre-configured for immediate simulation. |
| `re180_min_channel_1`               | This folder holds the output from a simulation run with a simulation time of 1. See the table below for details on key files and their configurations. |
| `longer_run_vtk`                    | Contains VTK files converted from specific ALYA output files (`mpio.run`) using the command: `mpirun -np 6 /scratch/polsm/alya_exe/mpio2vtk channel`. |
| `Plot_for_report`                   | This folder contains the images and related files that were used in the report. |
| `Gmsh`                   | This folder contains the Gmsh files for the meshing of the simulations. |

### Detailed Configuration of `re180_min_channel_1`

| File Name          | Parameter           | Description |
|--------------------|---------------------|-------------|
| `channel.dat`      | Timestep            | Set to 1. Use `mpirun -np 6 ./alya_3D.x channel` for execution, add `nohup` prefix for background processing. |
| `channel.ker.dat`  | Steps per output    | Configured for 50 steps before each output; tailored to maintain a Reynolds number of 180. |
| `channel.nsi.dat`  | Postprocess setting | Set to `VELOC` to save all velocity fields, allowing for detailed post-simulation analysis. |

## Key Files

| File Name          | Description |
|--------------------|-------------|
| `Re180.prof.txt`   | Data from Jiménez's paper used for comparison and validation of our simulations. |
| `requirements.txt` | Contains all the necessary packages required to run the programs in this project. |
| `ALYA_full_load_post_processing.ipynb` | This notebook allows loading the simulation at all times, enabling a study for each timestep. It is resource-intensive as it requires loading all timesteps into RAM. |
| `ALYA_small_load_post_processing.ipynb` | A lighter version of the previous notebook that only studies average statistics and compares them to Jiménez's results. |
| `ALYA_Q_event_Detection.ipynb` | This notebook focuses solely on the study and visualization of Q-events. |

## Jupyter Notebook Details

### ALYA_full_load_post_processing.ipynb
- **Purpose**: Load and analyze simulation data for each timestep.
- **Usage**: Suitable for detailed timestep-by-timestep analysis.

### ALYA_small_load_post_processing.ipynb
- **Purpose**: Study average statistics and compare with Jiménez's results.
- **Usage**: Efficient for statistical analysis without high memory usage.

### ALYA_Q_event_Detection.ipynb
- **Purpose**: Study and visualize Q-events in the simulation data.
- **Usage**: Focused on detecting, counting, and analyzing Q-events using percolation diagrams.

This README aims to clarify the structure and content of the ALYA repository for users, ensuring effective navigation and utilization of the provided simulation tools and data.
