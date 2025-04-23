#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <string>

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

#endif // DATA_GENERATOR_H
