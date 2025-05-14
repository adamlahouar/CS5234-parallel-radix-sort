#include <climits>
#include "serial_radix_sort.h"
#include "parallel_radix_sort.h"


int main() {
    constexpr int INPUT_SIZE = INT_MAX;
    constexpr int NUM_THREADS = 8;
    const auto inputArray = new int[INPUT_SIZE];
    const auto outputArray = new int[INPUT_SIZE];

    ParallelAllOpts::sort(inputArray, outputArray, INPUT_SIZE, NUM_THREADS);

    delete[] inputArray;
    delete[] outputArray;
}
