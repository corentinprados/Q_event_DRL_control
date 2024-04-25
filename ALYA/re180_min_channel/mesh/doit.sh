# load python environemnt
. /cfs/klemming/home/p/polsm/activate_tensorflow.sh 

# module load gmsh 
ml gmsh

gmsh -3 channel.geo

. /cfs/klemming/home/p/polsm/activate_pyalya.sh

srun -n 120 pyalya_gmsh2alya -p 3,4,5,6 -c channel channel

srun pyalya_periodic -c channel -d x,z --mpio channel

python3 initial_condition.py channel

cp *.mpio.bin ../.
