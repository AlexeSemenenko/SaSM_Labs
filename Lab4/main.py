import numpy as np
import scipy.integrate as integrate


rand = lambda: np.random.rand(1)[0]
n = 100000


def avg(sample):
    sample_sum = 0
    for val in sample:
        sample_sum += val

    return 1 / sample.size * sample_sum


def variance(sample):
    sqr_sum = 0
    mean = avg(sample)
    for val in sample:
        sqr_sum += (val - mean) ** 2

    return 1 / (sample.size - 1) * sqr_sum


def count_integral(sample):
    return avg(sample), 3 * np.sqrt(variance(sample) / sample.size)


# простой интеграл
class GaussGenerator:
    def __init__(self, m, s):
        self.m = m
        self.s = s

        self.N = 48

    def __get_standard__(self):
        return np.sqrt(12 / self.N) * (np.sum([rand() for _ in range(self.N)]) - self.N / 2)

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])

    def generate(self):
        return self.m + self.s * self.__get_standard__()


sample = GaussGenerator(0, 1).generate_sample(n)
transform_sample = np.sqrt(np.pi) / 2 * np.sqrt((2 + sample ** 2) / (np.sqrt(2) * np.abs(sample)))

computed_value = count_integral(transform_sample)
true_value = integrate.quad(lambda x: np.e ** (-x ** 4) * np.sqrt(1 + x ** 4), -np.inf, np.inf)
print(f'computed value: {computed_value[0]}, error: {computed_value[1]}')
print(f'true value: {true_value[0]}, error: {true_value[1]}')


# двойной интеграл
def generate_point():
    rand_x = 4 * rand() - 2
    rand_y = 4 * rand() - 2

    while not (1 <= rand_x ** 2 + rand_y ** 2 < 4):
        rand_x = 4 * rand() - 2
        rand_y = 4 * rand() - 2

    return 1 / (rand_x ** 2 + rand_y ** 4)


def generate_sample(size):
    return np.array([generate_point() for _ in range(size)])


sample = generate_sample(n)
square = 3 * np.pi
computed_value = avg(sample) * square

print(f'computed value: {computed_value}')
true_value = 3.857981450975978
print(f'\ntrue value: {true_value}')
