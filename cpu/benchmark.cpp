#include "serial_radix_sort.h"
#include "parallel_radix_sort.h"
#include "data_generator.h"

#include <functional>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <ranges>

constexpr int THREAD_COUNTS[] = {1, 2, 4, 8, 16, 32, 64};
constexpr int INPUT_SIZES[] = {
    2'000'000, 4'000'000, 8'000'000, 16'000'000, 32'000'000, 64'000'000, 128'000'000, 256'000'000, 512'000'000
};

constexpr DistributionType DISTRIBUTIONS[] = {
    DistributionType::UNIFORM,
    DistributionType::NORMAL,
    DistributionType::SKEW_SMALL,
    DistributionType::SKEW_LARGE
};

constexpr int NUM_RUNS = 7;

const std::string OUTPUT_FILENAME = "../cpu_benchmark_results.csv";
const std::string OUTPUT_COLUMNS = "Sorter,Input Distribution,Input Size,Thread Count,Average Execution Time [s]";

void runBenchmark(
    const std::string &sorterName,
    const std::function<void(int *, int *, int, int)> &sorter, // input, output, size, numThreads
    std::ofstream &outputFile,
    const int *originalData,
    const DistributionType distribution,
    const int numThreads,
    const int inputSize) {
    std::vector<long double> times(NUM_RUNS);

    for (int i = 0; i < NUM_RUNS; ++i) {
        const auto inputArray = new int[inputSize];
        const auto outputArray = new int[inputSize];
        std::memcpy(inputArray, originalData, sizeof(int) * inputSize);

        const auto start = std::chrono::high_resolution_clock::now();
        sorter(inputArray, outputArray, inputSize, numThreads);
        const auto end = std::chrono::high_resolution_clock::now();

        times[i] = std::chrono::duration<long double>(end - start).count();


        delete[] inputArray;
        delete[] outputArray;
    }

    std::ranges::sort(times);
    long double sum = 0.0;
    for (int i = 1; i < NUM_RUNS - 1; ++i) {
        sum += times[i];
    }
    const long double average = sum / (NUM_RUNS - 2);

    outputFile << std::fixed << std::setprecision(17)
            << sorterName << ","
            << DataGenerator::distToString(distribution) << ","
            << inputSize << ","
            << numThreads << ","
            << average << "\n";
}

int main() {
    std::ofstream outputFile(OUTPUT_FILENAME);
    outputFile << OUTPUT_COLUMNS << "\n";

    constexpr int MAX_INPUT_SIZE = INPUT_SIZES[std::size(INPUT_SIZES) - 1];
    std::unordered_map<DistributionType, int *> preGeneratedData;

    for (const auto distribution: DISTRIBUTIONS) {
        std::cout << "Generating max-size data for distribution: " << DataGenerator::distToString(distribution) << "\n";
        preGeneratedData[distribution] = DataGenerator::generate(MAX_INPUT_SIZE, distribution);
    }

    for (const auto inputSize: INPUT_SIZES) {
        std::cout << "Input size: " << inputSize << "\n";

        for (const auto distribution: DISTRIBUTIONS) {
            std::cout << "  Distribution: " << DataGenerator::distToString(distribution) << "\n";
            const int *originalData = preGeneratedData[distribution];

            std::cout << "    Running std::sort...\n";
            runBenchmark("std::sort", [&](int *input, int *, const int size, int) {
                std::sort(input, input + size);
            }, outputFile, originalData, distribution, 1, inputSize);

            std::cout << "    Running SerialRadixSort...\n";
            runBenchmark("SerialRadixSort", [&](int *input, int *, const int size, int) {
                SerialRadixSort::sort(input, size);
            }, outputFile, originalData, distribution, 1, inputSize);

            for (const auto numThreads: THREAD_COUNTS) {
                std::cout << "      Running BaseParallel with " << numThreads << " threads...\n";
                runBenchmark("BaseParallel", [&](const int *input, int *output, const int size, const int t) {
                    BaseParallel::sort(input, output, size, t);
                }, outputFile, originalData, distribution, numThreads, inputSize);

                std::cout << "      Running ParallelOptA with " << numThreads << " threads...\n";
                runBenchmark("ParallelOptA", [&](int *input, int *output, const int size, const int t) {
                    ParallelOptA::sort(input, output, size, t);
                }, outputFile, originalData, distribution, numThreads, inputSize);

                std::cout << "      Running ParallelOptB with " << numThreads << " threads...\n";
                runBenchmark("ParallelOptB", [&](const int *input, int *output, const int size, const int t) {
                    ParallelOptB::sort(input, output, size, t);
                }, outputFile, originalData, distribution, numThreads, inputSize);

                std::cout << "      Running ParallelOptC with " << numThreads << " threads...\n";
                runBenchmark("ParallelOptC", [&](const int *input, int *output, const int size, const int t) {
                    ParallelOptC::sort(input, output, size, t);
                }, outputFile, originalData, distribution, numThreads, inputSize);

                std::cout << "      Running ParallelOptAC with " << numThreads << " threads...\n";
                runBenchmark("ParallelOptAC", [&](int *input, int *output, const int size, const int t) {
                    ParallelOptAC::sort(input, output, size, t);
                }, outputFile, originalData, distribution, numThreads, inputSize);

                std::cout << "      Running ParallelAllOpts with " << numThreads << " threads...\n";
                runBenchmark("ParallelAllOpts", [&](int *input, int *output, const int size, const int t) {
                    ParallelAllOpts::sort(input, output, size, t);
                }, outputFile, originalData, distribution, numThreads, inputSize);
            }
        }
    }

    for (const auto &data: preGeneratedData | std::views::values) {
        delete[] data;
    }

    outputFile.close();
    std::cout << "Benchmark complete, results written to " << OUTPUT_FILENAME << "\n";

    return 0;
}
