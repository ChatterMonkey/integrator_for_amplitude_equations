import numpy as np
import matplotlib.pyplot as plt


eps = 0.01
dt = 0.0005
N = 10000


class trajectory_skeleton:
    def __init__(self, epsilon, initial_conditions, omega1, omega2 , q ,s ):
        self.epsilon = epsilon
        self.initial_conditions = initial_conditions
        self.omega1 = omega1
        self.omega2 = omega2
        self.q = q
        self.s = s

    def calculate(self, N, dt):
        data = compute_trajectory_data(self, N, dt)
        self.calculated_trajectory = [N, dt, data]

    def label(self):
        label = str(self.initial_conditions) + '_'  + str(self.omega2) + '_' + str(self.omega2) + '_' + str(self.epsilon) + '_' + str(self.q) + '_' + str(self.s)
        return label

    def legend(self):
        legend = self.label
        return legend

    def plot_rectangular(self, ax, N, dt):

        if hasattr(self, 'calculated_trajectory'):
            data = self.calculated_trajectory[2]
        else:
            self.calculate(self, N, dt)
            data = self.calculated_trajectory[2]

        print(data[:, 0])


        ax.plot(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]))




def f(trajectory_skeleton, A):
    A1, A2 = A
    print(A1)
    print(A2)
    w1 = trajectory_skeleton.omega1
    w2 = trajectory_skeleton.omega2
    s = trajectory_skeleton.s
    q = trajectory_skeleton.q

    array = np.zeros(2, dtype=np.complex64)

    array[0] = 1j * w1 * A1 + eps * A2 + 1j * A1 * (s[0, 0] * np.abs(A1) ** 2 + s[0, 1] * np.abs(A2) ** 2)
    array[1] = 1j * w2 * A2 + eps * q * A1 + 1j * A2 * (s[1, 0] * np.abs(A1) ** 2 + s[1, 1] * np.abs(A2) ** 2)
    return array


def compute_trajectory_data(trajectory_skeleton, N, dt):
    A10 = trajectory_skeleton.initial_conditions[0]
    A20 = trajectory_skeleton.initial_conditions[1]

    data = np.zeros((N, 2), dtype=np.complex64)
    data[0, :] = A10, A20
    for i in range(0, N - 1):
        print(i)
        print("DATRA")
        print(data[i, :])
        a1 = dt * f(trajectory_skeleton, data[i, :])
        a2 = dt * f(trajectory_skeleton, data[i, :] + a1 / 2)
        a3 = dt * f(trajectory_skeleton, data[i, :] + a2 / 2)
        a4 = dt * f(trajectory_skeleton, data[i, :] + a3)
        data[i + 1, :] = data[i, :] + (a1 + 2 * a2 + 2 * a3 + a4) / 6

    return data




t = trajectory_skeleton(0.01,[1,0],1,1,1,np.ones((2,2)))


#print(t.label())

t.calculate(10, 0.0005)
#print(t.calculated_trajectory)

ax = plt.figure().add_subplot(projection='3d')

t.plot_rectangular(ax, 10,  0.0005)

plt.show()
