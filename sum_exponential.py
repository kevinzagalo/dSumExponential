from numpy import prod, zeros, random
from math import exp


class SumExponential:

    def __init__(self, eta):
        self._eta = eta
        self._prod_eta = [prod([(eta_j - eta_i) for eta_j in eta[:i]+eta[i+1:]]) for i, eta_i in enumerate(eta)]
        self._weights = [prod([eta_j / (eta_j - eta_i) for eta_j in eta[:i]+eta[i+1:]]) for i, eta_i in enumerate(eta)]

    def pdf(self, x):
        return [prod(self._eta) * sum([exp(-eta_i * xx) / self._prod_eta[i]
                                      for i, eta_i in enumerate(self._eta)]) for xx in x]

    def cdf(self, x):
        return [sum([(1 - exp(-eta_j * xx)) * self._weights[j] for j, eta_j in enumerate(self._eta)]) for xx in x]

    @property
    def weights(self):
        return self._weights

    @property
    def params(self):
        return self._eta

    def sample(self, n_sample=1):
        S = zeros((n_sample, len(self._eta)))
        for i, eta_i in enumerate(self._eta):
            S[:, i] = random.exponential(1/eta_i, size=n_sample)
        return S.sum(axis=1)


if __name__ == '__main__':
    from numpy import linspace
    import matplotlib.pyplot as plt

    eta = [0.01, 0.2, 0.9, 0.0001]
    eta = list(set(eta))
    t_range = linspace(0.001, 1000)

    sum_exp = SumExponential(eta)
    print(sum_exp.sample(1000))
    plt.hist(sum_exp.sample(1000), bins=50, density=True)
    plt.plot(t_range, sum_exp.pdf(t_range))
    plt.show()

    print(sum_exp.params)
    print(sum(sum_exp.weights))
    plt.plot(t_range, sum_exp.cdf(t_range))
    plt.show()
