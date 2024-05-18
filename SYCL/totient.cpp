// ---------------------------------------------------
// ---------------- AUTHOR'S NAME---------------------

// SYCL Implementation done by Mael GRELLIER NEAU

// ---------------------------------------------------

#include <chrono>
#include <CL/sycl.hpp>
#include <array>
#include <iostream>

using namespace std::chrono;
using namespace cl::sycl;

class Totient;
class TotientReducer;
class TotientReducerLocality1D;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;
constexpr access::mode sycl_read_write = access::mode::read_write;

// Counter for arithmetic operations on long values
unsigned long long counter = 0;


long hcf(long x, long y){
  long t;

  while (y != 0) {
    t = x % y;
    x = y;
    y = t;

  }
  return x;
}

int relprime(long x, long y){
  return hcf(x, y) == 1;
}

long euler(long n){
  long length, i;

  length = 0;
  for (i = 1; i < n; i++)
    if (relprime(n, i))
      length++;
  return length;
}

long sumTotient(long lower, long upper){
  long sum, i;

  sum = 0;
  for (i = lower; i <= upper; i++)
    sum = sum + euler(i);
  return sum;
}


/**
 * SYCL Sequencial version
*/
void totientSequential(size_t rangeTotient){
  auto start = high_resolution_clock::now();
  double sequentialTotient = sumTotient(1, rangeTotient);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "sequential: " << sequentialTotient << ", time: " << duration.count() << " milli secs" << std::endl;
}

/**
 * SYCL version 1
 * sum on the cpu
*/
void totientSumCPU(size_t rangeTotient){
  // Because of the none constant input size, we are using a vector
  std::vector<cl_double> sums(rangeTotient);
  auto start = high_resolution_clock::now();

  { // beginning of SYCL objects scope, ensures data copied back to host
    queue deviceQueue;
    range<1> numOfItems{rangeTotient};
    buffer<cl_double, 1> bufferSums(sums.data(), numOfItems);
    deviceQueue.submit([&](handler& cgh) {
      auto accessorSums = bufferSums.template get_access<sycl_write>(cgh);
      auto kern = [=](id<1> wID) {
        accessorSums[wID[0]] = euler(((double) wID[0] + 1));
      };
      cgh.parallel_for<class Totient>(numOfItems, kern);
    });
  } // end of SYCL objects scope, ensures data copied back to host

  double sycl_1_result = 0.0;
  /* sum the values on the host CPU */
  for(unsigned int i=0; i<rangeTotient; i++) {
    sycl_1_result += sums[i];
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  
  std::cout << "SYCL v1: " << sycl_1_result << ", time: " << duration.count() << " milli secs" << std::endl; 
}


/**
 * SYCL version 2
 * use a reduction on the GPU device
*/
void totientReduction(size_t rangeTotient){
  unsigned int sycl_2_result = 0;
  auto start = high_resolution_clock::now();

  { // beginning of SYCL objects scope, ensures data copied back to host
    queue deviceQueue;
    range<1> numOfItems{rangeTotient};
    // Because of the none constant input size, we are using a vector
    std::vector<cl_double> sums(rangeTotient);
    buffer<cl_double, 1> bufferSums(sums.data(), numOfItems);
    buffer<unsigned int> bufferFinalSum { &sycl_2_result, 1 };
    try {
    deviceQueue.submit([&](handler& cgh) {
	/* reduction variable */
      auto sumReduction = reduction(bufferFinalSum, cgh, plus<unsigned int>());
      auto sum_kernel = [=](id<1> wID, auto& sum) {
        sum.combine(euler(((double) wID[0] + 1)));
      };
      cgh.parallel_for<class TotientReducer>(numOfItems, sumReduction, sum_kernel);
    });
    }
  catch(sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }

  } // end of SYCL objects scope, ensures data copied back to host

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "SYCL v2: " << sycl_2_result << ", time: " << duration.count() << " milli secs" << std::endl; 
}


/**
 * SYCL version 3
 * use explicit 1D workgroup size
*/
void totientWorkgroup(size_t rangeTotient, int workGroupSize){
  unsigned int sycl_3_result = 0;
  auto start = high_resolution_clock::now();

  { // beginning of SYCL objects scope, ensures data copied back to host
    queue deviceQueue;
    range<1> global_size{rangeTotient};
    range<1> my_work_group_size = workGroupSize;
    nd_range<1> numOfItems{global_size, my_work_group_size};
    // Because of the none constant input size, we are using a vector
    std::vector<cl_double> sums(rangeTotient);
    buffer<cl_double, 1> bufferSums(sums.data(), range<1>(rangeTotient));
    buffer<unsigned int> bufferFinalSum { &sycl_3_result, 1 };
    try {
      deviceQueue.submit([&](handler& cgh) {
	  /* reduction variable */
	  auto sumReduction = reduction(bufferFinalSum, cgh, plus<unsigned int>());
	  auto sum_kernel = [=](nd_item<1> item, auto& sum) {
        sum.combine(euler((double) item.get_global_id(0) + 1));

	  };
	  // this time defining work group size in first argument to parallel_for
	  cgh.parallel_for<class TotientReducerLocality1D>(numOfItems, sumReduction, sum_kernel);
	});
    }
    catch(sycl::exception const& e) {
      std::cout << "Caught synchronous SYCL exception:\n"
		<< e.what() << std::endl;
    }

  } // end of SYCL objects scope, ensures data copied back to host

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  // std::cout << "SYCL v3 for " << sycl_3_result << " a work group size of : "<< workGroupSize << ", time: " << duration.count() << " milli secs" << std::endl; 
  std::cout << "SYCL v3: " << sycl_3_result << ", time: " << duration.count() << " milli secs" << std::endl; 
 }



int main(int argc, char* argv[]) {
  std::cout.precision(10);

  /**
   * Manual version
  */


  const int rangeTotient = 100000;
  /**
   * SYCL Sequencial version
  */
  // totientSequential(rangeTotient);


  /**
   * SYCL version 1
   * sum on the cpu
  */
  totientSumCPU(rangeTotient);


  /**
   * SYCL version 2
   * use a reduction on the GPU device
  */
  totientReduction(rangeTotient);


  /**
   * SYCL version 3
   * use explicit 1D workgroup size
  */
  totientWorkgroup(rangeTotient, 500); 

  // sycl_profiler profiler(eventList, startList);
  // cout << "Kernel exection:\t" << profiler.get_kernel_execution_time() << endl;
  

  /**
   * Automatic tests
  */
  int tests[3] = {15000, 30000, 100000};
  // commun divider between 15000, 30000 and 100000
  int workgroups[14] = {1, 2, 4, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000};

  for (auto t : tests){
    std::cout << "------- Test for " << t << " -------" << std::endl;
    // totientSequential(t);
    totientSumCPU(t);
    totientReduction(t);

    for (auto wg : workgroups){
      std::cout << "-- Wg = " << wg << " --" << std::endl;
      totientWorkgroup(t, wg);
    }
    
  }
  


  /* all went well */
  return 0;
}
