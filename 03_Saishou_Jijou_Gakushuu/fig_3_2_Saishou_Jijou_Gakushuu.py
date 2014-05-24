
import numpy as np
import matplotlib.pyplot as plt

#http://gihyo.jp/dev/serial/01/machine-learning/0011
class Saishou_Jijou_Gakushuu_Gihyo:
    def __init__(self,
                 phi = lambda x: [1, x, x**2, x**3]):
        self._phi = phi
        self._plt = plt
        return
    """
    functions for learning
    """
    def get_phi_array(self, X):
        return np.array([self._phi(x) for x in X])
    def get_w(self, PHI, t):
        return np.dot(np.linalg.inv(np.dot(PHI.T, PHI)), np.dot(PHI.T, t))
    def learn(self,
              X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99]),
              t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.89, -0.79, -0.04])):
        self._plt.plot(X, t, 'o')
        PHI = self.get_phi_array(X)
        w = self.get_w(PHI, t)
        return w
    """
    functions for regression
    """
    def f(self, w, x):
        return np.dot(w, self._phi(x))
    def regression(self, w, xlist = np.arange(0, 1, 0.01)):
        ylist = [self.f(w, x) for x in xlist]
        self._plt.plot(xlist, ylist)
        return ylist
    def show_plot(self,):
        self._plt.show()

class Saishou_Jijou_Gakushuu:
    def __init__(self, lower = -3, upper = 3):
        self._lower = lower
        self._upper = upper
        return

    def get_x(self, n):
        return np.linspace(self._lower, self._upper, n)
    def get_large_X(self, large_N):
        return np.linspace(self._lower, self._upper, large_N)
    def get_pix(self, x):
        return np.pi * x
    def get_y(self, pix, x, n):
        # Matlab: y = sin(pix)./(pix) + 0.1*x + 0.05*randn(n,1)
        return np.sin(pix) / (pix) + 0.1*x + 0.05 * np.random.randn(n)

    def initialize_p(self, n, iteration):
        p = np.zeros( (n, 2*iteration) )
        # Matlab: p(:,1) = ones(n, 1) # n*1
        p[:,0] = np.ones(n)
        return p
    def initialize_large_P(self, large_N, iteration):
        P = np.zeros( (large_N, 2*iteration) )
        # Matlab: p(:,1) = ones(N, 1) # N*1
        P[:,0] = np.ones(large_N)
        return P
    def append_p(self, p, j, x):
        p[:, 2*j]   = np.sin(j/2*x)
        p[:, 2*j+1] = np.cos(j/2*x)
        return p
    def append_large_P(self, large_P, j, large_X):
        large_P[:, 2*j]   = np.sin(j/2*large_X)
        large_P[:, 2*j+1] = np.cos(j/2*large_X)
        return large_P

    def get_t(self, p, y):
        #x = np.linalg.solve(p, y)
        x,resid,rank,s = np.linalg.lstsq(p, y)
        return x
    def get_large_F(self, large_P, t):
        return np.dot(large_P, t)

    def plot(self, large_X, large_F, x, y):
        plt.plot(large_X, large_F)
        plt.plot(x, y, 'o')
        plt.show()
        return

    def process(self, n, large_N, iteration):
        x = self.get_x(n)
        large_X = self.get_large_X(large_N)
        pix = self.get_pix(x)
        y = self.get_y(pix, x, n)

        # initialize p, P
        p = self.initialize_p(n, iteration)
        large_P = self.initialize_large_P(large_N, iteration)
        # loop on p, P
        for j in range(iteration):
            p = self.append_p(p, j, x)
            large_P = self.append_large_P(large_P, j, large_X)
        t = self.get_t(p, y)
        large_F = self.get_large_F(large_P, t)

        # plot
        self.plot(large_X, large_F, x, y)
        return

def process_gihyo_with_default_data():
    # initialize with phi
    SJG = Saishou_Jijou_Gakushuu_Gihyo()
    # learn with default training data
    w = SJG.learn()
    # regression
    ylist = SJG.regression(w)
    SJG.show_plot()
def process_gihyo(test_mode = False):
    if test_mode == True:
        process_gihyo_with_default_data()
        return
    phi = lambda x: [1, np.sin(x/2), np.cos(x/2),
                     np.sin(2*x/2), np.cos(2*x/2),
                     np.sin(3*x/2), np.cos(3*x/2),
                     np.sin(4*x/2), np.cos(4*x/2),
                     np.sin(5*x/2), np.cos(5*x/2)] # TODO: extend this list to [np.sin(15*x/2), np.cos(15*x/2)]
    # initialize with phi
    SJG = Saishou_Jijou_Gakushuu_Gihyo(phi)
    # training data
    n = 50
    X = np.linspace(-3, 3, n)
    t = np.sin(np.pi * X) / (np.pi * X) + 0.1*X + 0.05 * np.random.randn()
    # learn
    w = SJG.learn(X, t)
    # regression
    N = 1000
    xlist = np.linspace(-3, 3, N)
    ylist = SJG.regression(w, xlist)
    SJG.show_plot()
def process():
    SJG = Saishou_Jijou_Gakushuu()
    SJG.process(n=50, large_N=1000, iteration=15)
def main(use_gihyo_version = False):
    if use_gihyo_version:
        process_gihyo()
    else:
        process()

if __name__ == '__main__':
    main()
