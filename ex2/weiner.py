import numpy as np
from numpy.linalg import inv

class WeinerDenoiseFilter:

    def __init__(self, mean, cov, noise_std):
        self.dim = len(mean)
        self.mean = mean
        self.inv_cov = inv(cov)
        self.noise_std_squred = noise_std ** 2
        self.left_operand = inv(self.inv_cov + np.identity(self.dim) / self.noise_std_squred)

    def __call__(self, *args, **kwargs):
        if len(args) > 1: raise Exception("Unkwon args")
        img = args[0]
        right_operand = self.inv_cov @ self.mean + img / self.noise_std_squred
        return self.left_operand @ right_operand
