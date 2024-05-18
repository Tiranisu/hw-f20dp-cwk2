# ---------------------------------------------------
# ---------------- AUTHOR'S NAME---------------------

# MPI Reduce Benchmark done by Enzo Peigné

# ---------------------------------------------------

# If you are collecting data for the coursework, you should run this
# script on the head node on the cluster.
# This script is done to get the runtime of each configuration
# Better to comment each lines in TotientRange.c used for the counter incrementation
echo "input,cores,run1,run2,run3,run4,run5" > runtimeReduce.csv

for inputSize in 15000 30000 100000
do
    for cores in 1 2 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192
    do
        echo -n $inputSize >> runtimeReduce.csv
        echo -n "," >> runtimeReduce.csv
        echo -n $cores >> runtimeReduce.csv
        for k in 1 2 3 4 5
        do
            echo -n "," >> runtimeReduce.csv
            # to test on any computer (e.g. in EM 2.50 or your laptop)
            # printf "%s" "$(mpirun -n 4 ./totientMPIReduce 1 $inputSize)" >> runtimeReduce.csv
            module load intel/mpi/64
            # to test on a Robotarium cluster compute node
            printf "%s" "$(srun --partition=amd-longq --mpi=pmi2  -N 3 -n $cores ./totientMPIReduce 1 $inputSize)" >> runtimeReduce.csv
        done
        echo "" >> runtimeReduce.csv
    done
done
