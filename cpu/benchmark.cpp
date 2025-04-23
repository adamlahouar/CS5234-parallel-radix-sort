#include "serial_radix_sort.h"
#include "parallel_radix_sort.h"
#include "data_generator.h"

#include <string>
#include <fstream>
#include <iostream>
#include <functional>
#include <cstring>
#include <omp.h>
#include <algorithm>

constexpr int THREAD_COUNTS[] = {1, 2, 4, 8, 16, 32, 64};
constexpr int INPUT_SIZES[] = {
    2'000, 4'000, 8'000, 16'000, 32'000, 64'000, 128'000, 256'000, 512'000
};

constexpr DistributionType DISTRIBUTIONS[] = {
    DistributionType::UNIFORM,
    DistributionType::NORMAL,
    DistributionType::SKEW_SMALL,
    DistributionType::SKEW_LARGE
};

constexpr int NUM_RUNS = 7;

const std::string OUTPUT_FILENAME = "../benchmark_results.csv";
const std::string OUTPUT_COLUMNS = "Sorter,Input Distribution,Input Size,Thread Count,Average Execution Time [s]";


void runBenchmark(
    const std::string &sorterName,
    const std::function<void(int *, int, int)> &sorter, // sorter(arr, size, numThreads)
    std::ofstream &outputFile,
    const int *originalData,
    const DistributionType distribution,
    const int numThreads,
    const int inputSize) {

    std::vector<double> times(NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; ++i) {
        const auto data = new int[inputSize];
        std::memcpy(data, originalData, sizeof(int) * inputSize);

        const auto start = omp_get_wtime();
        sorter(data, inputSize, numThreads);
        const auto end = omp_get_wtime();

        times[i] = end - start;
        delete[] data;
    }

    std::ranges::sort(times);
    double sum = 0.0;
    for (int i = 1; i < NUM_RUNS - 1; ++i) {
        sum += times[i];
    }
    const double average = sum / (NUM_RUNS - 2);

    outputFile << sorterName << ","
            << DataGenerator::distToString(distribution) << ","
            << inputSize << ","
            << numThreads << ","
            << average << "\n";
}

int main() {
    std::ofstream outputFile(OUTPUT_FILENAME);
    outputFile << OUTPUT_COLUMNS << "\n";

    for (const auto inputSize: INPUT_SIZES) {
        std::cout << "Input size: " << inputSize << "\n";

        for (const auto distribution: DISTRIBUTIONS) {
            std::cout << "  Distribution: " << DataGenerator::distToString(distribution) << "\n";

            const int *originalData = DataGenerator::generate(inputSize, distribution);

            std::cout << "    Running std::sort..." << "\n";
            runBenchmark("std::sort", [&](int *data, const int size, int) { std::sort(data, data + size); },
                         outputFile, originalData, distribution, 1, inputSize);

            std::cout << "    Running serial radix sort..." << "\n";
            runBenchmark("SerialRadixSort", [&](int *data, const int size, int) {
                SerialRadixSort::sort(data, size);
            }, outputFile, originalData, distribution, 1, inputSize);

            for (const auto numThreads: THREAD_COUNTS) {
                std::cout << "      Running parallel radix sort with " << numThreads << " threads..." << "\n";
                runBenchmark("ParallelRadixSort", [&](int *data, const int size, const int t) {
                    ParallelRadixSort::sort(data, size, t);
                }, outputFile, originalData, distribution, numThreads, inputSize);
            }

            delete[] originalData;
        }
    }

    outputFile.close();
    std::cout << "Benchmark complete, results written to " << OUTPUT_FILENAME << "\n";

    return 0;
}
