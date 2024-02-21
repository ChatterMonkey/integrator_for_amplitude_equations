import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
from os.path import exists

from matplotlib.patches import Rectangle

eps = 0.01
dt = 0.0005
N = 100

def label_maker(trajectory_skeleton):
    A1 = str(trajectory_skeleton.initial_conditions[0])
    A2 = str(trajectory_skeleton.initial_conditions[1])
    omega1 = str(trajectory_skeleton.omega1)
    omega2 = str(trajectory_skeleton.omega2)
    q = str(trajectory_skeleton.q )
    epsilon = str(trajectory_skeleton.epsilon)
    s = trajectory_skeleton.s
    s00 = str(s[0][0])
    s01 = str(s[0][1])
    s10 = str(s[1][0])
    s11 = str(s[1][1])
    label = A1 + '_' + A2 + '_' + omega1 + '_' + omega2 + '_' + epsilon + '_' + q + '_' + s00 + '_' + s01 + '_' + s10 + '_' + s11
    return label


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
    def has_data(self):
        return hasattr(self, 'calculated_trajectory')
    def label(self):
        #label = str(self.initial_conditions) + '_'  + str(self.omega2) + '_' + str(self.omega2) + '_' + str(self.epsilon) + '_' + str(self.q) + '_' + str(self.s)
        return label_maker(self)
    def legend(self):
        legend = self.label
        return legend
    def serialize(self):
        file_name = self.label() + ".txt"
        with open(file_name, "wb") as file:
            pickle.dump(self,file)
    def plot(self, ax, N, dt, basis):
        if hasattr(self, 'calculated_trajectory'):
            data = self.calculated_trajectory[2]
        else:
            self.calculate(self, N, dt)
            data = self.calculated_trajectory[2]
        basis(ax,data)

def rectangular(ax, data):
    ax.plot(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]))

def rectangular2(ax, data):
    ax.plot(np.imag(data[:, 1]), np.real(data[:, 0]), np.real(data[:, 1]))

def f(trajectory_skeleton, A):
    A1, A2 = A
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
        a1 = dt * f(trajectory_skeleton, data[i, :])
        a2 = dt * f(trajectory_skeleton, data[i, :] + a1 / 2)
        a3 = dt * f(trajectory_skeleton, data[i, :] + a2 / 2)
        a4 = dt * f(trajectory_skeleton, data[i, :] + a3)
        data[i + 1, :] = data[i, :] + (a1 + 2 * a2 + 2 * a3 + a4) / 6

    return data



def procure(epsilon, initial_conditions, omega1, omega2, q, s, N, dt):
    t = trajectory_skeleton(epsilon, initial_conditions,omega1, omega2,q,s)
    label = t.label()
    filename = label + ".txt"
    if os.path.exists(filename):
        print("Ressurecting...")
        object = pickle.load(open(filename, "rb"))
        if object.has_data():
            return object
        else:
            object.calculate(N,dt)
            return object
    else:
        print("Building from scratch...")
        t.calculate(N,dt)
        print("Built")
        return t

def display_together(objects, basis):
    ax = plt.figure().add_subplot(projection='3d')
    for object in objects:
        N = object.calculated_trajectory[0]
        dt = object.calculated_trajectory[1]
        object.plot( ax, N, dt, basis)
    plt.show()




t = procure(0.0005, [0,1], 1,1,1,np.ones((2,2)), 10, 0.001)
t.serialize()
h = procure(0.0005, [1,1], 1,1,1,np.ones((2,2)), 10, 0.001)
h.serialize()
print(h.label())


display_together([t,h], rectangular2)
