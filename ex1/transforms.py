import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., magnitude=1):
        self.std = std
        self.mean = mean
        self.magnitude = magnitude

    def __call__(self, item):
        tensor = item[0]
        return (tensor + torch.randn(tensor.size()) * (self.std + self.mean) * self.magnitude, item[1])

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)
