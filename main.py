from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt


eps = 0.01
dt = 0.0005
N = 10000
trajdata = []



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

        ax.plot(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]))







def f(trajectory_skeleton, A):
    A1, A2 = A
    w1 = trajectory_skeleton.omega1
    w2 = trajectory_skeleton.omega2
    s = trajectory_skeleton.s
    q = trajectory_skeleton.q

    array = np.zeros(2, dtype=np.complex64)

    array[0] = 1j * w1 * A1 + eps * A2 + 1j * A1 * (s[0, 0] * np.abs(A1) ** 2 + s[0, 1] * np.abs(A2) ** 2)
    array[1] = 1j * w2 * A2 + eps * q* A1 + 1j * A2 * (s[1, 0] * np.abs(A1) ** 2 + s[1, 1] * np.abs(A2) ** 2)
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




t = trajectory_skeleton(0.01,[1,0],1,1,1,np.ones((2,2)))


#print(t.label())

#t.calculate(10000, 0.0005)
#print(t.calculated_trajectory)

#ax = plt.figure().add_subplot(projection='3d')

#t.plot_rectangular(ax, 10000,  0.0005)

#plt.show()



























def F(A, omega, eps):
    A1, A2 = A

    array = np.zeros(2, dtype=np.complex64)
    s = np.ones((2, 2))
    array[0] = 1j * omega[0] * A1 + eps * A2 + 1j * A1 * (s[0, 0] * np.abs(A1) ** 2 + s[0, 1] * np.abs(A2) ** 2)
    array[1] = 1j * omega[1] * A2 + eps * A1 + 1j * A2 * (s[1, 0] * np.abs(A1) ** 2 + s[1, 1] * np.abs(A2) ** 2)
    return array

def compute_traj(
        trajdata):  # calculates and stores trajectories, given a list of their data, returns dictionary of trajectories
    trajs = dict()
    for [A1, A2, omega, eps, dt, N] in trajdata:

        traj = np.zeros((N, 2), dtype=np.complex64)
        traj[0, :] = A1, A2
        for i in range(0, N - 1):
            a1 = dt * F(traj[i, :], omega, eps)
            a2 = dt * F(traj[i, :] + a1 / 2, omega, eps)
            a3 = dt * F(traj[i, :] + a2 / 2, omega, eps)
            a4 = dt * F(traj[i, :] + a3, omega, eps)
            traj[i + 1, :] = traj[i, :] + (a1 + 2 * a2 + 2 * a3 + a4) / 6

        name = label(A1, A2, omega, eps, dt, N)
        trajs[name] = traj
        np.save('trajectories/' + name, traj)
        print("Computed " + name + ":")
        print(traj[0:3])
        print("...")
        print(traj[N - 3:N])
    return trajs





#each row is re(A1), im(A1), re(A2), im(A2)
#initial_components = np.array([[1, 0, 1, 0], [np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)],[0, 1, 0, 1]])


initial_components = np.array([[1,0,0,0]])


omegas = np.array([[1,1]])
legends = ["A1, A2 = 1, omega1, omega2 = 1,2","A1, A2 =2 omega1, omega2 = 1,2","A1, A2 =3 omega1, omega2 = 1,2"]
#print(initial_components.shape[0])


for run in range(0,initial_components.shape[0]):
    A1  = initial_components[run,0] + initial_components[run,1]*1j
    A2 = initial_components[run, 2] + initial_components[run, 3] * 1j
    trajdata.append([A1, A2, omegas[run], eps, dt, N])

#print(trajdata)

def label(A1, A2, omega, eps, dt, N):
    return (str(A1) + '_' + str(A2) + '_' + str(omega[0]) + '_' + str(omega[1]) + '_' + str(eps) + '_' + str(dt) + '_' + str(N) + '.npy')
def unlable(string):
    stringettes = string.strip(".npy").split("_")
    for i in range(0,len(stringettes)):
        if i in [0,1]:
            stringettes[i] = complex(stringettes[i])
        else:
            stringettes[i] = float(stringettes[i])
    return stringettes



def show_rectangular_phasespace(N, dt, trajs, legends):
    ax = plt.figure().add_subplot(projection='3d')
    values = list(trajs.values())
    i = 0
    for trajectory in values:

        ax.plot(np.imag(trajectory[:, 0]), np.real(trajectory[:, 0]), np.real(trajectory[:, 1]), label = legends[i] )
        i = i+1
    plt.legend()

def show_parabolic_phasespace(N, dt, trajs, legends):

    ax = plt.figure().add_subplot(projection='3d')
    values = list(trajs.values())
    i = 0
    for trajectory in values:
        A1 = trajectory[: ,0]
        A2 = trajectory[: ,1]
        z1 = 1j * A1
        z2 = np.conjugate(A2)
        hype_trajs = np.zeros((trajectory.shape[0], 3) )
        print(hype_trajs.shape)
        hype_trajs[:, 0] = np.abs(A1)**2
        hype_trajs[:, 1] = np.cos(np.angle(A1) + np.angle(A2))
        hype_trajs[:, 2] = np.cos(np.angle(A2))
        print(trajectory)
        print(hype_trajs)
        ax.plot(hype_trajs[:, 0], hype_trajs[:, 1], hype_trajs[:, 2], label=legends[i])
        plt.xlabel("|A1|")
        plt.ylabel("cos(phi + phi')")
        plt.ylabel("cos( phi')")
        i = i + 1
    plt.legend()


    plt.show()
def show_components(N, dt, trajs):
    ax = plt.figure().add_subplot()
    time = np.arange(N) * dt
    values = list(trajs.values())
    for trajectory in values:
        ax.plot(time, np.real(trajectory[:, 0]))

    plt.show()

def reconstitute(trajdata):
    trajs = {}
    for data in trajdata:
        name = label(*data)
        trajs[name] = np.load("trajectories/" + name)

    return trajs
def hamiltonian(A1, A2, w1, w2, eps):
    z1 = 1j * A1
    z2 = np.conjugate(A2)

    h = 0.5 * (w2 * np.abs(z2) ** 2 - w1 * np.abs(z1) ** 2) - eps * np.real(z1 * z2) - 0.25 * np.abs(
        z1) ** 4 - 0.5 * np.abs(z1 * z2) ** 2 + 0.25 * np.abs(z2) ** 4

    return h

def angularish(A1, A2):
    z1 = 1j * A1
    z2 = np.conjugate(A2)

    j = 0.5*(np.abs(z1)**2 - np.abs(z2)**2)
    return j




def extract_values(trajs, samples):

    keys = list(trajs.keys())
    for key in keys:
        data = unlable(key)
        w1 = data[2]
        w2 = data[3]
        eps = data[4]
        trajectory = trajs[key]
        for sample in samples:
            A1 =trajectory[sample][0]
            A2 = trajectory[sample][1]
            h = hamiltonian(A1, A2, w1, w2, eps)
            j = angularish(A1, A2)
            print("HAMILTONIANH IS " + str(h))
            print("J IS " + str(j))


    return


N = 10000
dt = 0.0005

trajs = compute_traj(trajdata)
#trajs = reconstitute(trajdata)
#extract_values(trajs,[0,10,100,5000])


#show_parabolic_phasespace(N,dt, trajs, legends)
show_rectangular_phasespace(N, dt, trajs, legends)
#show_components(N,dt, trajs)



# RK4
