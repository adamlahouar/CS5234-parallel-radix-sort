#include <climits>

#include "data_generator.h"
#include "parallel_radix_sort.h"


int main() {
    constexpr int n = INT_MAX;
    const auto data = DataGenerator::generate(n, DistributionType::NORMAL);

    constexpr int num_threads = 8;

    ParallelRadixSort::sort(data, n, num_threads);
}
