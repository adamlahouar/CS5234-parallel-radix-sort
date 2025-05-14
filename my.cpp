
#include <string>
#include <thread>
#include <omp.h>
#include <random>
#include <algorithm>

enum class DistributionType {
    UNIFORM,
    NORMAL,
    SKEW_SMALL,
    SKEW_LARGE
};

class DataGenerator {
public:
    static int *generate(int size, DistributionType distType);

    static std::string distToString(DistributionType distribution);
};




constexpr int UNIFORM_MIN = 0;
constexpr int UNIFORM_MAX = 1'000'000;

constexpr double NORMAL_MEAN = 500'000;
constexpr double NORMAL_STD = 200'000;

constexpr double GAMMA_SHAPE = 2.0;
constexpr double GAMMA_SCALE = 100'000;
constexpr int GAMMA_MAX = 1'000'000;

constexpr int BASE_SEED = 42;
const int MAX_THREADS = std::thread::hardware_concurrency();

int *DataGenerator::generate(const int size, const DistributionType distType) {
    const auto data = new int[size];

    omp_set_num_threads(MAX_THREADS);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        std::mt19937 rng(BASE_SEED + tid);

        switch (distType) {
            case DistributionType::UNIFORM: {
                std::uniform_int_distribution dis(UNIFORM_MIN, UNIFORM_MAX);
                #pragma omp for
                for (std::size_t i = 0; i < size; ++i)
                    data[i] = dis(rng);
                break;
            }

            case DistributionType::NORMAL: {
                std::normal_distribution dis(NORMAL_MEAN, NORMAL_STD);
                #pragma omp for
                for (std::size_t i = 0; i < size; ++i)
                    data[i] = std::clamp(static_cast<int>(dis(rng)), UNIFORM_MIN, UNIFORM_MAX);
                break;
            }

            case DistributionType::SKEW_SMALL: {
                std::gamma_distribution dis(GAMMA_SHAPE, GAMMA_SCALE);
                #pragma omp for
                for (std::size_t i = 0; i < size; ++i)
                    data[i] = std::min(static_cast<int>(dis(rng)), GAMMA_MAX);
                break;
            }

            case DistributionType::SKEW_LARGE: {
                std::gamma_distribution dis(GAMMA_SHAPE, GAMMA_SCALE);
                #pragma omp for
                for (std::size_t i = 0; i < size; ++i)
                    data[i] = std::max(GAMMA_MAX - static_cast<int>(dis(rng)), UNIFORM_MIN);
                break;
            }
        }
    }

    return data;
}

std::string DataGenerator::distToString(const DistributionType distribution) {
    switch (distribution) {
        case DistributionType::UNIFORM: return "Uniform";
        case DistributionType::NORMAL: return "Normal";
        case DistributionType::SKEW_SMALL: return "Skew Small";
        case DistributionType::SKEW_LARGE: return "Skew Large";
    }
    return "Unknown";
}
