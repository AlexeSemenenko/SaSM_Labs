import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
import math


# значение базовой случайной величины
rand = lambda: np.random.rand(1)[0]
# количество значений случайной величины
n = 10000


# центральный момент
def center_moment(sample, deg):
    return 1 / sample.size * ((sample - avg(sample)) ** deg).sum()


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


# коэффицент корреляции
def corr(feat1, feat2):
    m1, m2 = avg(feat1), avg(feat2)
    s1, s2 = np.sqrt(variance(feat1)), np.sqrt(variance(feat2))

    return 1 / feat1.size * np.sum((feat1 - m1) * (feat2 - m2) / s1 / s2)


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


# плотность распределения
def plot_pdf(left_bound, right_bound, pdf, pdf_params):
    x = np.linspace(left_bound, right_bound, 10000)
    y = np.vectorize(pdf)(x, *pdf_params)

    plt.plot(x, y, color='lightskyblue', label='true_pdf')
    plt.legend()


# критерий хи-квадрат Пирсона
def chi2(sample_size, estimated_probs, true_probs, eps):
    statistic = sample_size * ((estimated_probs - true_probs) ** 2 / true_probs).sum()
    bound = sts.chi2.ppf(1 - eps, estimated_probs.size - 1)

    return 0 if statistic < bound else 1, statistic, bound, eps


print('Gauss')


# нормальное распределение
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


m, s = 1, 3
sample = GaussGenerator(m, s).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: {m}')
print(f'estimated variance: {variance(sample)}', f'true variance: {s ** 2}')

bin_count = 7
probs = get_probs(sample, sts.norm.cdf, [m, s], bin_count)

min_, max_ = sample.min(), sample.max()
plot_bars(min_, max_, probs['estimated'], probs['true'], 'Gauss')
plt.show()
plot_pdf(min_, max_, sts.norm.pdf, [m, s])
plt.show()

print(f'chi2: ', chi2(n, probs['estimated'], probs['true'], 0.05))
#


print('\nLogNorm')


# логнормальное распределение
class LogNormGenerator:
    def __init__(self, m, s):
        self.m = m
        self.s = s

        self.N = 48

    def __get_standard__(self):
        return np.sqrt(12 / self.N) * (np.sum([rand() for _ in range(self.N)]) - self.N / 2)

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])

    def generate(self):
        return math.exp(self.m + self.s * self.__get_standard__())


# функция распределения
def lognorm_cdf(x, m, s):
    return 0.5 * math.erfc(-(math.log(x) - m) / (s * 2 ** 0.5))


# плотность
def lognorm_pdf(x, m, s):
    return (1 / (x * s * (2 * math.pi) ** 0.5)) * math.exp(-(math.log(x) - m) ** 2 / (2 * s ** 2))


m, s = 1, 3
sample = LogNormGenerator(m, s).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: {math.exp(m + (s ** 2) / 2)}')
print(f'estimated variance: {variance(sample)}', f'true variance: {(math.exp(s ** 2) - 1) * math.exp(2 * m + s ** 2)}')

bin_count = 7
probs = get_probs(sample, lognorm_cdf, [m, s], bin_count)

min_, max_ = sample.min(), sample.max()
plot_bars(min_, max_, probs['estimated'], probs['true'], 'LogNorm')
plt.show()
plot_pdf(min_, max_, lognorm_pdf, [m, s])
plt.show()

print(f'chi2: ', chi2(n, probs['estimated'], probs['true'], 0.05))
#


print('\nExp')


# экспоненциальное распределение
class ExpGenerator:
    def __init__(self, a):
        self.a = a

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])

    def generate(self):
        return - 1 / self.a * math.log(rand())


# функция распределения
def exp_cdf(x, a):
    return 1 - math.exp(- a * x)


# плотность
def exp_pdf(x, a):
    return a * math.exp(- a * x)


a = 2
sample = ExpGenerator(a).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: {1 / a}')
print(f'estimated variance: {variance(sample)}', f'true variance: {1 / a ** 2}')

bin_count = 10
probs = get_probs(sample, exp_cdf, [a], bin_count)

min_, max_ = sample.min(), sample.max()
plot_bars(min_, max_, probs['estimated'], probs['true'], 'Exp')
plt.show()
plot_pdf(min_, max_, exp_pdf, [a])
plt.show()

print(f'chi2: ', chi2(n, probs['estimated'], probs['true'], 0.05))
#


print('\nLaplace')


# распределение Лапласа
class LaplaceGenerator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def generate_sample(self, size):
        r = [rand() for _ in range(size + 1)]
        return np.array([self.generate(r[i], r[i + 1]) for i in range(size)])

    def generate(self, r1, r2):
        return self.a + (1 / self.b) * math.log(r1 / r2)


# функция распределения
def laplace_cdf(x, a, b):
    return 0.5 * math.exp(a * (x - b)) if x <= b else 1 - 0.5 * math.exp(-a * (x - b))


# плотность
def laplace_pdf(x, a, b):
    return a / 2 * math.exp(- a * math.fabs(x - b))


a, b = 0.5, 0.5
sample = LaplaceGenerator(a, b).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: {a}')
print(f'estimated variance: {variance(sample)}', f'true variance: {2 / b ** 2}')

bin_count = 7
probs = get_probs(sample, laplace_cdf, [a, b], bin_count)

min_, max_ = sample.min(), sample.max()
plot_bars(min_, max_, probs['estimated'], probs['true'], 'Laplace')
plt.show()
plot_pdf(min_, max_, laplace_pdf, [a, b])
plt.show()

print(f'chi2: ', chi2(n, probs['estimated'], probs['true'], 0.05))
#


print('\nWeibull')


# распределение Вейбулла
class WeibullGenerator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])

    def generate(self):
        return self.a * math.pow(-math.log(rand()), 1 / self.b)


a, b = 1, 0.5
sample = WeibullGenerator(a, b).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: {a * math.gamma((1 / b) + 1)}')
print(f'estimated variance: {variance(sample)}', f'true variance: {(a ** 2) * (math.gamma((2 / b) + 1) - math.pow(math.gamma((1 / b) + 1), 2))}')

bin_count = 5
probs = get_probs(sample, sts.exponweib.cdf, [a, b], bin_count)

min_, max_ = sample.min(), sample.max()
plot_bars(min_, max_, probs['estimated'], probs['true'], 'Weibull')
plt.show()
plot_pdf(min_, max_, sts.exponweib.pdf, [a, b])
plt.show()

print(f'chi2: ', chi2(n, probs['estimated'], probs['true'], 0.05))
#


print('\nMix')


# логнормальное + экспоненциальное распределение
class MixGenerator:
    def __init__(self, pi, lognorm_params, exp_params):
        self.pi = pi
        self.generators = [LogNormGenerator(*lognorm_params), ExpGenerator(*exp_params)]

    def get_distribution(self, prob):
        return int(prob <= 1 - self.pi) if self.pi > 0.5 else 1 - (prob <= self.pi)

    def generate_sample(self, size):
        return np.array([self.generate() for _ in range(size)])

    def generate(self):
        return self.generators[self.get_distribution(rand())].generate()


lognorm, exp, pi = [1, 3], [2], 0.4
sample = MixGenerator(pi, lognorm, exp).generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: '
                                       f'{pi * math.exp(lognorm[0] + (lognorm[1] ** 2) / 2) + (1 - pi) * (1 / exp[0])}')
# d = pi * d1 + (1-pi) * d2 + pi * (1-pi)* (m1-m2)^2
print(f'estimated variance: {variance(sample)}', f'true variance: {pi * (math.exp(lognorm[1] ** 2) - 1) * math.exp(2 * lognorm[0] + lognorm[1] ** 2) + (1 - pi) * (1 / exp[0] ** 2) + pi * (1 - pi) * (math.exp(lognorm[0] + (lognorm[1] ** 2) / 2) - (1 / exp[0])) ** 2}')
#


print('\nBox-Muller')


# нормальное распределение с преобразованием Бокса-Мюллера
class BoxMullerGenerator:
    def generate(self):
        radius, angle = np.sqrt(-2 * np.log(rand())), 2 * np.pi * rand()
        return radius * np.sin(angle), radius * np.cos(angle)

    def generate_sample(self, size):
        sample = []

        for _ in range(size // 2):
            gauss = self.generate()
            sample.append(gauss[0])
            sample.append(gauss[1])

        return np.array(sample)


sample = BoxMullerGenerator().generate_sample(n)

print(f'estimated avg: {avg(sample)}', f'true avg: {0}')
print(f'estimated variance: {variance(sample)}', f'true variance: {1}')

bin_count = 7
probs = get_probs(sample, sts.norm.cdf, [0, 1], bin_count)

min_, max_ = sample.min(), sample.max()
plot_bars(min_, max_, probs['estimated'], probs['true'], 'Box-Muller')
plt.show()
plot_pdf(min_, max_, sts.norm.pdf, [0, 1])
plt.show()

print(f'corr: {corr(sample[::2], sample[1::2])}')
#
