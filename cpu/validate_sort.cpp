#include "parallel_radix_sort.h"
#include "data_generator.h"

#include <iostream>
#include <cstring>
#include <algorithm>
#include <locale>

constexpr int INPUT_SIZE = 100'000'000;
constexpr auto DISTRIBUTION = DistributionType::NORMAL;
constexpr int NUM_THREADS = 8;

bool isValid(const int *arr, const int *expected, const int n) {
    for (int i = 0; i < n; ++i) {
        if (arr[i] != expected[i]) {
            std::cout << "Mismatch at index " << i << ": " << arr[i] << " != " << expected[i] << "\n";
            return false;
        }
    }

    std::cout << "Sorted array is valid.\n";
    return true;
}

int main() {
    std::cout << "Validating ParallelRadixSort...\n";
    std::cout << "- Thread count: " << NUM_THREADS << "\n";
    std::cout << "- Input size:   " << INPUT_SIZE << "\n";
    std::cout << "- Distribution: " << DataGenerator::distToString(DISTRIBUTION) << "\n\n";

    const auto *originalData = DataGenerator::generate(INPUT_SIZE, DISTRIBUTION);

    const auto expectedData = new int[INPUT_SIZE];
    std::memcpy(expectedData, originalData, sizeof(int) * INPUT_SIZE);
    std::sort(expectedData, expectedData + INPUT_SIZE);

    const auto parallelData = new int[INPUT_SIZE];
    std::memcpy(parallelData, originalData, sizeof(int) * INPUT_SIZE);
    ParallelRadixSort::sort(parallelData, INPUT_SIZE, NUM_THREADS);

    const auto result = isValid(parallelData, expectedData, INPUT_SIZE);

    delete[] originalData;
    delete[] expectedData;
    delete[] parallelData;

    return !result;
}
