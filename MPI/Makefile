# ---------------------------------------------------
# ---------------- AUTHOR'S NAME---------------------

# MPI Makefile done by Enzo Peigné

# ---------------------------------------------------


CORES = 64 # Define the number of cores to run the MPI program on

# Compile the MPI program
all:
	mpicc -o totientMPI TotientRange.c
	mpicc -o totientMPIReduce TotientReduce.c

# Run the MPI program on 4 cores on local machine
exec:
	mpirun -n 4 ./totientMPI 1 15000
	mpirun -n 4 ./totientMPI 1 30000
	mpirun -n 4 ./totientMPI 1 100000

# Run the MPI program on X cores on the Robotarium cluster
run:
	srun --partition=amd-longq --mpi=pmi2  -N 2 -n $(CORES) ./totientMPI 1 15000
	srun --partition=amd-longq --mpi=pmi2  -N 2 -n $(CORES) ./totientMPI 1 30000
	srun --partition=amd-longq --mpi=pmi2  -N 2 -n $(CORES) ./totientMPI 1 100000

# Clean the directory
clean:
	rm -f totientMPI
	rm -f totientMPIReduce
