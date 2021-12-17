class LC:
    def __init__(self, x, a, c, m):
        self.x = x
        self.a = a
        self.c = c
        self.m = m

    def generate(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x


class MM:
    def __init__(self, lc1, lc2, k, n):
        self.lc1 = lc1
        self.lc2 = lc2
        self.m = self.lc2.m
        self.k = k
        self.n = n

        self.V = [lc1.x, *[self.lc1.generate() for _ in range(k - 1)]]

    def generate(self):
        j = int(self.lc2.x * self.k / self.m)
        z = self.V[j]
        self.V[j] = self.lc1.x

        self.lc1.generate()
        self.lc2.generate()

        return z
