class parSet:
    def __init__(self, dim, walk, num_walk, q, p):
        self.dim = dim
        self.walk = walk
        self.num_walk = num_walk
        self.q = q
        self.p = p
    def __str__(self):
        return str(self.dim) + '_' + str(self.walk) + '_' + str(self.num_walk) + '_' + str(self.p) + '_' + str(self.q)