
import numpy as np
import matplotlib.pyplot as plt

class Saishou_Jijou_Gakushuu:
    def __init__(self,):
        return

    def get_x(self, n, lower = -3, upper = 3):
        return np.linspace(lower, upper, n)
    def get_X(self, N, lower = -3, upper = 3):
        return np.linspace(lower, upper, N)
    def get_pix(self, x):
        return np.pi * x
    def get_y(self, pix, x, n):
        return np.sin(pix) / pix + 0.1*x + 0.05 + np.random.randn(n, 1)

    def initialize_p(self, n, iteration):
        p = np.zeros( (n, 2*iteration) )
        # p(:,1) = ones(n, 1) # n*1
        p[:,1] = np.ones(n)
        return p
    def initialize_P(self, N, iteration):
        P = np.zeros( (N, 2*iteration) )
        # p(:,1) = ones(N, 1) # N*1
        P[:,1] = np.ones(N)
        return P
    def append_p(self, p, j, x):
        p[:, 2*j]   = np.sin(j/2*x)
        p[:, 2*j+1] = np.cos(j/2*x)
        return p
    def append_P(self, P, j, X):
        P[:, 2*j]   = np.sin(j/2*X)
        P[:, 2*j+1] = np.cos(j/2*X)
        return P

    def get_t(self, p, y):
        return p / y
    def get_F(self, P, t):
        return P * t

    def plot(self, x, y):
        plt.plot(x, y)
        return

    def process(self, n, large_N, iteration):
        x = self.get_x(n)
        large_X = self.get_X(large_N)
        pix = self.get_pix(x)
        y = self.get_y(pix, x, n)

        # initialize p, P
        p = self.initialize_p(n, iteration)
        large_P = self.initialize_P(large_N, iteration)
        # loop on p, P
        for j in range(iteration):
            p = self.append_p(p, j, x)
            large_P = self.append_P(large_P, j, large_X)
        t = self.get_t(p, y)
        large_F = self.get_F(large_P, t)

        self.plot(large_X, large_F)
        self.plot(x,y)
        return

def main():
    SJG = Saishou_Jijou_Gakushuu()
    SJG.process(n=50, large_N=1000, iteration=15)

if __name__ == '__main__':
    main()
