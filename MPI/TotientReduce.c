#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// ---------------------------------------------------
// ---------------- AUTHOR'S NAME---------------------

// MPI Reduce Implementation done by Enzo Peigné

// ---------------------------------------------------

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
  return sum;
}

int main(int argc, char **argv)
{
  long lower, upper, local_sum = 0, global_sum;
  int id, p;
  double elapsedTime;

  // Initialize MPI VM
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Check if the number of arguments is correct
  if (argc != 3) {
    if (id == 0)
      printf("Usage: %s lower_num upper_num\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  sscanf(argv[1], "%ld", &lower);
  sscanf(argv[2], "%ld", &upper);
 
  // Start the timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (id==0)    elapsedTime = - MPI_Wtime();


  // Calculate local sum
  for (long i = lower + id; i <= upper; i += p) {
    local_sum += euler(i);
    counter++;
  }

  // Reduce local sums to get global sum
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (id == 0) {
    elapsedTime += MPI_Wtime();
    // printf("Sum of Totients between [%ld..%ld] is %ld\n", lower, upper, global_sum);
    // Benchmark Time Print:
    printf("%f", elapsedTime);

  
    // long frequency = counter/elapsedTime/1000000; // MHz

    // printf("Number of long operations: %lld\n", counter);
    // printf("Frequency: %ld MHz\n", frequency);

  }

  MPI_Finalize(); // Shut down MPI VM
  return 0;
}

