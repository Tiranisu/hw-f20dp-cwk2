#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// ---------------------------------------------------
// ---------------- AUTHOR'S NAME---------------------

// MPI Implementation done by Enzo Peigné

// ---------------------------------------------------

// Define the maximum number of processors
#define MAX_PROCS 192

// Counter for the number of operations
unsigned long long counter = 0; 

long hcf(long x, long y)
{
  long t;

  while (y != 0) {
    t = x % y;
    x = y;
    y = t;
    counter++; // Count the number of operations
  }
  return x;
}

int relprime(long x, long y)
{
  return hcf(x, y) == 1;
}

long euler(long n)
{
  long length, i;

  length = 0;
  for (i = 1; i < n; i++)
    if (relprime(n, i))
      length++;
    counter++; // Count the number of operations
  return length;
}

long sumTotient(long lower, long upper)
{
  long sum = 0, i;

  for (i = lower; i <= upper; i++)
    sum += euler(i);
    counter++; // Count the number of operations
  return sum;
}

int main(int argc, char **argv)
{
  long lower, upper;
  int p;                                // Size of VM
  int id;                               // Rank in VM
  double elapsedTime;
  double seqTime, parTime;
  float speedup, efficiency;
  
  MPI_Init(&argc, &argv);               // start up "virtual machine"
  MPI_Comm_size(MPI_COMM_WORLD, &p);    // get size of VM
  MPI_Comm_rank(MPI_COMM_WORLD, &id);   // get own rank in VM

  // Check if the number of arguments is correct
  if (argc != 3) {
    if (id == 0) {
      printf("Usage: %s lower upper\n", argv[0]);
    }
    MPI_Finalize();
    exit(1);
  }

  sscanf(argv[1], "%ld", &lower);
  sscanf(argv[2], "%ld", &upper);

  // Sequential version
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (id == 0)    elapsedTime = - MPI_Wtime();

  // if (id == 0) {
    // printf("Sum of Totients from %ld to %ld is %ld\n", lower, upper, sumTotient(lower, upper));
    // elapsedTime += MPI_Wtime();
    // printf("Time: %f seconds\n", elapsedTime);
    // seqTime = elapsedTime;
  // }

  // Parallel version
  char name[80];
  int z, i;
  int num_threads;
  unsigned long sum; // sum of Totients
  unsigned long sums[MAX_PROCS]; // sum buffer
  MPI_Status status;

  num_threads = p;
  
  // Start the timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (id==0)    elapsedTime = - MPI_Wtime();

  // Calculate local sum
  sum = 0l;
  for(i=id; i<=upper; i+=p) {
    sum += euler(i);
    counter++;
  }

  // Store the local sums in the buffer and send them to the master process
  if (id==0) {
    sums[0] = sum;
  } else {
    MPI_Send(&sum, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);        // send the result back to the master
  }
    
  // Master collects the results and calculates the global sum
  if (id==0) {		     
    for(i = 1; i < p; i++) {
      MPI_Recv(&sums[i], 1, MPI_LONG, i, 0, MPI_COMM_WORLD, &status);
      sum = (sum + sums[i]);
    }
  }
  
  // Print the result
  if (id==0) {
    elapsedTime += MPI_Wtime();
    // printf("Sum of Totients from %ld to %ld is %ld\n", lower, upper, sum);
    printf("%f", elapsedTime);
    parTime = elapsedTime;

    // Display speedup and efficiency
    // speedup = seqTime/parTime;
    // efficiency = speedup/p;    
    // printf("Speedup: %f\n", speedup);
    // printf("Efficiency: %f\n", efficiency);

    // long frequency = counter/elapsedTime/1000000; // in MHz

    // printf("Number of long operations: %lld\n", counter);
    // printf("Frequency: %ld MHz\n", frequency);

  }
  
  MPI_Finalize();                       // shut down VM
  return 0;
}
