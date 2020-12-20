def plot_hopping(self, T):
    i, j = T
    v = self.pos_matrix[i] - self.pos_matrix[j]
    if np.sqrt(v[0]**2 + v[1]**2) < 1.1:
        x = [self.pos_matrix[i, 0], self.pos_matrix[j, 0]]
        y = [self.pos_matrix[i, 1], self.pos_matrix[j, 1]]
        plt.plot(x, y, color="black")

def plot(self, show=True):
    if self.lattice_type == "graphene":
        plt.figure(figsize=(10, 6))
    else:
        plt.figure(figsize=(10, 10))

    X = self.pos_matrix[:, 0]
    Y = self.pos_matrix[:, 1]
    plt.scatter(X, Y, color="black")

    it = itertools.product(self.sub_matrix[0], self.sub_matrix[1])
    list(map(self.plot_hopping, it))

    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1, np.amax(X) + 1)
    plt.ylim(-1, np.amax(Y) + 1)

    if show:
        plt.show()
