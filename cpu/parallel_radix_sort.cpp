#include "parallel_radix_sort.h"

#include <omp.h>
#include <utility>
#include <algorithm>
#include <cstring>

constexpr int BITS_PER_PASS = 8;
constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

void ParallelRadixSort::sort(int *arr, const int n, const int numThreads) {
    omp_set_num_threads(numThreads);
    auto buffer = new int[n];

    for (int shift = 0; shift < sizeof(int) * 8; shift += BITS_PER_PASS) {
        const auto localHistograms = new int *[numThreads];
        const auto globalHistogram = new int[NUM_BUCKETS]{};
        const auto prefixSums = new int[NUM_BUCKETS]{};
        const auto threadOffsets = new int *[numThreads];

        for (int t = 0; t < numThreads; ++t) {
            localHistograms[t] = new int[NUM_BUCKETS]{};
            threadOffsets[t] = new int[NUM_BUCKETS]{};
        }

        #pragma omp parallel default(none) shared(arr, buffer, localHistograms, globalHistogram, prefixSums, threadOffsets, shift, n, numThreads)
        {
            const int tid = omp_get_thread_num();

            computeLocalHistograms(arr, n, localHistograms[tid], shift);
            #pragma omp barrier

            #pragma omp single
            {
                computeGlobalHistogram(localHistograms, globalHistogram, numThreads);
                computePrefixSums(globalHistogram, prefixSums);
                computeThreadOffsets(localHistograms, prefixSums, threadOffsets, numThreads);
            }
            #pragma omp barrier

            scatterToBuffer(arr, n, buffer, threadOffsets[tid], shift);
        }

        std::swap(arr, buffer);

        for (int t = 0; t < numThreads; ++t) {
            delete[] localHistograms[t];
            delete[] threadOffsets[t];
        }

        delete[] localHistograms;
        delete[] globalHistogram;
        delete[] prefixSums;
        delete[] threadOffsets;
    }

    delete[] buffer;
}

void ParallelRadixSort::computeLocalHistograms(const int *__restrict arr, const int n, int *__restrict localHistograms,
                                               const int shift) {
    std::memset(localHistograms, 0, NUM_BUCKETS * sizeof(int));
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
        localHistograms[bucket]++;
    }
}


void ParallelRadixSort::computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
    std::memset(globalHistogram, 0, NUM_BUCKETS * sizeof(int));
    for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
        for (int t = 0; t < numThreads; ++t) {
            globalHistogram[bucket] += localHistograms[t][bucket];
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
        for (int t = 0; t < numThreads; ++t) {
            threadOffsets[t][bucket] = offset;
            offset += localHistograms[t][bucket];
        }
    }
}

void ParallelRadixSort::scatterToBuffer(const int *arr, const int n, int *buffer, int *threadOffsets, const int shift) {
    int privateOffsets[NUM_BUCKETS];
    std::memcpy(privateOffsets, threadOffsets, NUM_BUCKETS * sizeof(int));

    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int value = arr[i];
        const int bucket = (value >> shift) & (NUM_BUCKETS - 1);
        buffer[privateOffsets[bucket]++] = value;
    }

    std::memcpy(threadOffsets, privateOffsets, NUM_BUCKETS * sizeof(int));
}
