#include "serial_radix_sort.h"

#include <utility>
#include <cstring>

constexpr int BITS_PER_PASS = 1;
constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

void SerialRadixSort::sort(int *arr, const int n) {
    auto buffer = new int[n];

    for (int shift = 0; shift < sizeof(int) * 8; shift += BITS_PER_PASS) {
        int histogram[NUM_BUCKETS] = {};
        buildHistogram(arr, n, histogram, shift);

        int prefixSums[NUM_BUCKETS] = {};
        computePrefixSums(histogram, prefixSums);

        scatterToBuffer(arr, n, buffer, prefixSums, shift);

        std::swap(arr, buffer);
    }

    delete[] buffer;
}


void SerialRadixSort::buildHistogram(const int *arr, const int n, int *histogram, const int shift) {
    std::memset(histogram, 0, sizeof(int) * NUM_BUCKETS);

    for (int i = 0; i < n; i++) {
        const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
        histogram[bucket]++;
    }
}

void SerialRadixSort::computePrefixSums(const int *histogram, int *prefixSums) {
    int sum = 0;
    for (int bucket = 0; bucket < NUM_BUCKETS; bucket++) {
        prefixSums[bucket] = sum;
        sum += histogram[bucket];
    }
}

void SerialRadixSort::scatterToBuffer(const int *arr, const int n, int *buffer, int *prefixSums, const int shift) {
    for (int i = 0; i < n; i++) {
        const int value = arr[i];
        const int bucket = (value >> shift) & (NUM_BUCKETS - 1);
        const int pos = prefixSums[bucket]++;
        buffer[pos] = value;
    }
}
