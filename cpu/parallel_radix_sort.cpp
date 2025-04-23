#include "parallel_radix_sort.h"

#include <omp.h>
#include <utility>

constexpr int BITS_PER_PASS = 8;
constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

void ParallelRadixSort::sort(int *arr, const int n, const int numThreads) {
    omp_set_num_threads(numThreads);

    auto buffer = new int[n];

    for (int shift = 0; shift < sizeof(int) * 8; shift += BITS_PER_PASS) {
        auto **localHistograms = new int *[numThreads];
        for (int thread = 0; thread < numThreads; ++thread) {
            localHistograms[thread] = new int[NUM_BUCKETS]{};
        }

        computeLocalHistograms(arr, n, localHistograms, shift);

        auto *globalHistogram = new int[NUM_BUCKETS]{};
        computeGlobalHistogram(localHistograms, globalHistogram, numThreads);

        auto *prefixSums = new int[NUM_BUCKETS]{};
        computePrefixSums(globalHistogram, prefixSums);

        auto **threadOffsets = new int *[numThreads];
        for (int thread = 0; thread < numThreads; ++thread) {
            threadOffsets[thread] = new int[NUM_BUCKETS]{};
        }
        computeThreadOffsets(localHistograms, prefixSums, threadOffsets, numThreads);

        scatterToBuffer(arr, n, buffer, threadOffsets, shift);

        std::swap(arr, buffer);

        for (int thread = 0; thread < numThreads; ++thread) {
            delete[] localHistograms[thread];
            delete[] threadOffsets[thread];
        }

        delete[] localHistograms;
        delete[] threadOffsets;
        delete[] globalHistogram;
        delete[] prefixSums;
    }

    delete[] buffer;
}

void ParallelRadixSort::computeLocalHistograms(const int *arr, const int n, int **localHistograms, const int shift) {
    #pragma omp parallel default(none) shared(arr, n, localHistograms, shift)
    {
        const int tid = omp_get_thread_num();
        int *local = localHistograms[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
            local[bucket]++;
        }
    }
}

void ParallelRadixSort::computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
    for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
        for (int thread = 0; thread < numThreads; ++thread) {
            globalHistogram[bucket] += localHistograms[thread][bucket];
        }
    }
}

void ParallelRadixSort::computePrefixSums(const int *globalHistogram, int *prefixSums) {
    int sum = 0;
    for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
        prefixSums[bucket] = sum;
        sum += globalHistogram[bucket];
    }
}

void ParallelRadixSort::computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets,
                                             const int numThreads) {
    for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
        int offset = prefixSums[bucket];

        for (int thread = 0; thread < numThreads; ++thread) {
            threadOffsets[thread][bucket] = offset;
            offset += localHistograms[thread][bucket];
        }
    }
}


void ParallelRadixSort::scatterToBuffer(const int *arr, const int n, int *buffer, int **threadOffsets,
                                        const int shift) {
    #pragma omp parallel default(none) shared(arr, n, buffer, threadOffsets, shift)
    {
        const int tid = omp_get_thread_num();
        int *local = threadOffsets[tid];

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            const int value = arr[i];
            const int bucket = (value >> shift) & (NUM_BUCKETS - 1);
            const int pos = local[bucket]++;
            buffer[pos] = value;
        }
    }
}
