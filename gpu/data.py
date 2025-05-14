import numpy as np
from enum import Enum

# Constants
UNIFORM_MIN = 0
UNIFORM_MAX = 1_000_000

NORMAL_MEAN = 500_000
NORMAL_STD = 200_000

GAMMA_SHAPE = 2.0
GAMMA_SCALE = 100_000
GAMMA_MAX = 1_000_000

BASE_SEED = 42

class DistributionType(Enum):
    UNIFORM = 1
    NORMAL = 2
    SKEW_SMALL = 3
    SKEW_LARGE = 4

class DataGenerator:

    @staticmethod
    def generate(size, dist_type):
        rng = np.random.default_rng(BASE_SEED)
        if dist_type == DistributionType.UNIFORM:
            data = rng.integers(UNIFORM_MIN, UNIFORM_MAX + 1, size=size)
        elif dist_type == DistributionType.NORMAL:
            samples = rng.normal(NORMAL_MEAN, NORMAL_STD, size=size)
            data = np.clip(samples.astype(int), UNIFORM_MIN, UNIFORM_MAX)
        elif dist_type == DistributionType.SKEW_SMALL:
            samples = rng.gamma(GAMMA_SHAPE, GAMMA_SCALE, size=size)
            data = np.minimum(samples.astype(int), GAMMA_MAX)
        elif dist_type == DistributionType.SKEW_LARGE:
            samples = rng.gamma(GAMMA_SHAPE, GAMMA_SCALE, size=size)
            data = np.maximum(GAMMA_MAX - samples.astype(int), UNIFORM_MIN)
        else:
            raise ValueError("Unknown distribution type")

        return data

    @staticmethod
    def dist_to_string(distribution):
        if distribution == DistributionType.UNIFORM:
            return "Uniform"
        elif distribution == DistributionType.NORMAL:
            return "Normal"
        elif distribution == DistributionType.SKEW_SMALL:
            return "Skew Small"
        elif distribution == DistributionType.SKEW_LARGE:
            return "Skew Large"
        else:
            return "Unknown"

# Example usage:
if __name__ == "__main__":
    size = 10_000
    dist_type = DistributionType.NORMAL

    generated_data = DataGenerator.generate(size, dist_type)
    print(f"First 10 generated numbers ({DataGenerator.dist_to_string(dist_type)}): {generated_data[:10]}")