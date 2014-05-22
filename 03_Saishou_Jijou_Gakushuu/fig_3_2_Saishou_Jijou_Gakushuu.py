
import numpy

class Saishou_Jijou_Gakushuu:
    def __init__():
        return

    def get_x(self, n):
        return
    def get_X(self, N):
        return
    def get_pix(x):
        return
    def get_y(self, pix, x, n):
        return
    def get_p(self, j, x):
        return
    def get_P(self, j, x):
        return
    def get_t(self, p, y):
        return
    def get_F(self, P, t):
        return
    def plot(self, X, F, x, y):
        return

    def process(n, N, iter):
        x = self.get_x(n)
        X = self.get_X(N)
        pix = self.get_pix(x)
        y = self.get_y(pix, x, n)
        for j in range(iter):
            p = self.get_p(j, x)
            P = self.get_P(j, X)
        t = get_t(p, y)
        F = get_F(P, t)

        self.plot(X, F, x, y)
        return


