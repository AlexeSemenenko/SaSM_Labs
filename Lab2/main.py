import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

# значение базовой случайной величины
rand = lambda: np.random.rand(1)[0]
# количество значений случайной величины
n = 10000


# центральный момент
def center_moment(sample, deg):
    return 1 / sample.size * ((sample - sample.mean()) ** deg).sum()


# мат ожидание
def avg(sample):
    sample_sum = 0
    for val in sample:
        sample_sum += val

    return 1 / sample.size * sample_sum


# дисперсия
def variance(sample):
    sqr_sum = 0
    mean = avg(sample)
    for val in sample:
        sqr_sum += (val - mean) ** 2

    return 1 / (sample.size - 1) * sqr_sum


# коэффицент эксцесса
def kurtosis(sample):
    n = sample.size

    return (n ** 2 - 1) / (n - 2) / (n - 3) * (center_moment(sample, 4) / center_moment(sample, 2) ** 2 - 3 + 6 / (n + 1))


# коэффицент асимметрии
def skewness(sample):
    n = sample.size

    return (n ** 2 - n) ** 0.5 / (n - 2) * center_moment(sample, 3) / center_moment(sample, 2) ** 1.5


# практические и теоритические вероятности
def get_probs(sample, cdf, cdf_params, bin_count):
    min_val, max_val = sample.min(), sample.max()
    bin_size = (max_val - min_val) / bin_count

    probs = np.array([cdf(min_val + i * bin_size, *cdf_params) for i in range(bin_count + 1)])
    true_probs = probs[1:] - probs[:-1]
    true_probs[0] += probs[0]
    true_probs[-1] += 1 - cdf(max_val, *cdf_params)

    estimated_probs = np.array([np.sum((min_val + i * bin_size < sample) & (sample <= min_val + (i + 1) * bin_size))
                                for i in range(bin_count)]) / sample.size
    estimated_probs[0] += (sample == min_val).sum() / sample.size

    return {'estimated': estimated_probs, 'true': true_probs}


# гистограмма
def plot_bars(left_bound, right_bound, estimated_probs, true_probs, label):
    width = (right_bound - left_bound) / estimated_probs.size
    bins = np.linspace(left_bound, right_bound, estimated_probs.size + 1)

    plt.bar(bins[:-1], estimated_probs, width=width, color='green', label='estimated', align='edge')
    plt.bar(bins[:-1], true_probs, width=width, color='lightskyblue', label='true', align='edge')
    plt.legend()
    plt.title(label)


# график эмпирической и теоритической функция распределения
def plot_cdf(sample, cdf, cdf_params, label):
    sample_values = np.unique(sample)

    estimated_values = np.array([np.sum(sample <= value) for value in sample_values]) / sample.size
    true_values = np.array([cdf(value, *cdf_params) for value in sample_values])

    plt.hlines(estimated_values[:-1], sample_values[:-1], sample_values[1:], color='green', label='estimated')
    plt.hlines(true_values[:-1], sample_values[:-1], sample_values[1:], color='lightskyblue', label='true')
    plt.legend()
    plt.title(label)


# Бернулли генератор
class BernoulliGenerator:
    def __init__(self, p):
        self.p = p

    def generate(self):
        return 1 if rand() <= self.p else 0

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])


p = 0.3
sample = BernoulliGenerator(p).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: {p}')
print(f'estimated variance: {variance(sample)}', f'true variance: {p * (1 - p)}')
print(f'estimated kurtosis: {kurtosis(sample)}', f'true kurtosis: {(6 * p ** 2 - 6 * p + 1) / p / (1 - p)}')
print(f'estimated skewness: {skewness(sample)}', f'true skewness: {(1 - 2 * p) / (p - p ** 2) ** 0.5}')

bin_count = 2
probs = get_probs(sample, sts.bernoulli.cdf, [p], bin_count)

plot_bars(sample.min(), sample.max(), probs['estimated'], probs['true'], 'Bernoulli')
plt.show()

plot_cdf(sample, sts.bernoulli.cdf, [p], 'Bernoulli')
plt.show()
#

print('')


# Геометрический генератор
class GeometricGenerator:
    def __init__(self, p):
        self.p = p
        self.bern_generator = BernoulliGenerator(self.p)

    def generate(self):
        n = 1
        while not self.bern_generator.generate():
            n += 1

        return n

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])


p = 0.2
sample = GeometricGenerator(p).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true mean: {1 / p}')
print(f'estimated variance: {variance(sample)}', f'true variance: {(1 - p) / p ** 2}')
print(f'estimated kurtosis: {kurtosis(sample)}', f'true kurtosis: {6 + p ** 2 / (1 - p)}')
print(f'estimated skewness: {skewness(sample)}', f'true skewness: {(2 - p) / np.sqrt(1 - p)}')

bin_count = 9
probs = get_probs(sample, sts.geom.cdf, [p], bin_count)

plot_bars(sample.min(), sample.max(), probs['estimated'], probs['true'], 'Geometric')
plt.show()

plot_cdf(sample, sts.geom.cdf, [p], 'Geometric')
plt.show()
#

print('')


# Биномиальный генератор
class BinomialGenerator:
    def __init__(self, m, p):
        self.m = m
        self.p = p
        self.bern_generator = BernoulliGenerator(self.p)

    def generate(self):
        return np.sum([self.bern_generator.generate() for _ in range(self.m)])

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])


m, p = 5, 0.6
sample = BinomialGenerator(m, p).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true mean: {m * p}')
print(f'estimated variance: {variance(sample)}', f'true variance: {m * p * (1 - p)}')
print(f'estimated kurtosis: {kurtosis(sample)}', f'true kurtosis: {(1 - 6 * (p - p ** 2)) / (m * (p - p ** 2))}')
print(f'estimated skewness: {skewness(sample)}', f'true skewness: {(1 - 2 * p) / np.sqrt(m * (p - p ** 2))}')

bin_count = 5
probs = get_probs(sample, sts.binom.cdf, [m, p], bin_count)

plot_bars(sample.min(), sample.max(), probs['estimated'], probs['true'], 'Binomial')
plt.show()

plot_cdf(sample, sts.binom.cdf, [m, p], 'Binomial')
plt.show()
#

print('')


# Обратный биномиальный генератор
class InverseBinomialGenerator:
    def __init__(self, r, p):
        self.r = r
        self.p = p
        self.bern_generator = BernoulliGenerator(self.p)

    def generate(self):
        success_count = 0
        fail_count = 0

        while success_count < self.r:
            if self.bern_generator.generate():
                success_count += 1
            else:
                fail_count += 1

        return fail_count

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])


r, p = 6, 0.25
sample = InverseBinomialGenerator(r, p).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true mean: {r * (1 - p) / p}')
print(f'estimated variance: {variance(sample)}', f'true variance: {r * (1 - p) / p ** 2}')
print(f'estimated kurtosis: {kurtosis(sample)}', f'true kurtosis: {6 / r + p ** 2 / r / (1 - p)}')
print(f'estimated skewness: {skewness(sample)}', f'true skewness: {(2 - p) / np.sqrt(r * (1 - p))}')

bin_count = 8
probs = get_probs(sample, sts.nbinom.cdf, [r, p], bin_count)

plot_bars(sample.min(), sample.max(), probs['estimated'], probs['true'], 'Inverse Binomial')
plt.show()

plot_cdf(sample, sts.nbinom.cdf, [r, p], 'Inverse Binomial')
plt.show()
#
