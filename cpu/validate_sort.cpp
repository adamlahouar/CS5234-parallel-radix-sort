#include "parallel_radix_sort.h"
#include "data_generator.h"

#include <iostream>
#include <cstring>
#include <algorithm>

constexpr int INPUT_SIZE = 8'000'000;
constexpr auto DISTRIBUTION = DistributionType::NORMAL;
constexpr int NUM_THREADS = 8;

bool isValid(const int *arr, const int *expected, const int n) {
    for (int i = 0; i < n; ++i) {
        if (arr[i] != expected[i]) {
            std::cout << "  Mismatch at index " << i << ": " << arr[i] << " != " << expected[i] << "\n";
            return false;
        }
    }

    std::cout << "  Sorted array is valid.\n";
    return true;
}

int main() {
    std::cout << "Validating ParallelRadixSort implementations...\n";
    std::cout << "- Thread count: " << NUM_THREADS << "\n";
    std::cout << "- Input size:   " << INPUT_SIZE << "\n";
    std::cout << "- Distribution: " << DataGenerator::distToString(DISTRIBUTION) << "\n\n";

    auto originalData = DataGenerator::generate(INPUT_SIZE, DISTRIBUTION);

    const auto expectedData = new int[INPUT_SIZE];
    std::memcpy(expectedData, originalData, sizeof(int) * INPUT_SIZE);
    std::sort(expectedData, expectedData + INPUT_SIZE);

    auto validateSort = [&](const std::string &name, auto sortFunction) -> bool {
        auto *output = new int[INPUT_SIZE];
        std::memcpy(output, originalData, sizeof(int) * INPUT_SIZE);

        std::cout << "Testing " << name << "...\n";
        sortFunction(originalData, output, INPUT_SIZE, NUM_THREADS);
        const bool valid = isValid(output, expectedData, INPUT_SIZE);

        if (!valid) {
            std::cout << "  " << name << " failed validation.\n";
        }

        delete[] output;
        return valid;
    };

    bool allValid = true;

    allValid &= validateSort("BaseParallel::sort", BaseParallel::sort);
    allValid &= validateSort("ParallelOptA::sort", ParallelOptA::sort);
    allValid &= validateSort("ParallelOptB::sort", ParallelOptB::sort);
    allValid &= validateSort("ParallelOptC::sort", ParallelOptC::sort);
    allValid &= validateSort("ParallelOptAC::sort", ParallelOptAC::sort);
    allValid &= validateSort("ParallelAllOpts::sort", ParallelAllOpts::sort);

    delete[] originalData;
    delete[] expectedData;

    if (!allValid) {
        std::cout << "\nSome implementations failed validation.\n";
        return 1;
    }

    std::cout << "\nAll implementations passed validation.\n";
    return 0;
}
