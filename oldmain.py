
import numpy as np
import matplotlib.pyplot as plt

w1 = 1
w2  = 1
eps = 0.01
dt = 0.0005
N = 10000
trajdata = []

#each row is re(A1), im(A1), re(A2), im(A2)
initial_components = np.array([[1, 0, 0, 0]])

print(initial_components.shape[0])


for run in range(0,initial_components.shape[0]):
    A1  = initial_components[run,0] + initial_components[run,1]*1j
    A2 = initial_components[run, 2] + initial_components[run, 3] * 1j
    trajdata.append([A1, A2, w1,w2, eps, dt, N])

print(trajdata)

def label(A1, A2, w1,w2, eps, dt, N):
    return ('traj_(' + str(A1) + ',' + str(A2) + ')_' + str(w1) + str(w2) + '_' + str(eps) + '_' + str(dt) + '_' + str(N) + '.npy')

def f(A, w1,w2, eps):
    A1, A2 = A

    array = np.zeros(2, dtype=np.complex64)
    s = np.ones((2, 2))
    array[0] = 1j * w1 * A1 + eps * A2 + 1j * A1 * (s[0, 0] * np.abs(A1) ** 2 + s[0, 1] * np.abs(A2) ** 2)
    array[1] =   1j * w2 * A2 + eps * A1 + 1j * A2 * (s[1, 0] * np.abs(A1) ** 2 + s[1, 1] * np.abs(A2) ** 2)
    return array


def compute_traj(trajdata):  # calculates and stores trajectories, given a list of their data, returns dictionary of trajectories
    trajs = dict()
    for [A1, A2, w1,w2, eps, dt, N] in trajdata:

        traj = np.zeros((N, 2), dtype=np.complex64)
        traj[0, :] = A1, A2
        for i in range(0, N - 1):
            a1 = dt * f(traj[i, :], w1,w2, eps)
            a2 = dt * f(traj[i, :] + a1 / 2, w1,w2, eps)
            a3 = dt * f(traj[i, :] + a2 / 2, w1,w2, eps)
            a4 = dt * f(traj[i, :] + a3, w1,w2, eps)
            traj[i + 1, :] = traj[i, :] + (a1 + 2 * a2 + 2 * a3 + a4) / 6

        name = label(A1, A2, w1,w2, eps, dt, N)
        trajs[name] = traj
        np.save('trajectories/' + name, traj)
        print("Computed " + name + ":")
        print(traj[0:3])
        print("...")
        print(traj[N - 3:N])
    return trajs

def show_phasespace(N, dt, trajs):
    ax = plt.figure().add_subplot(projection='3d')
    values = list(trajs.values())
    for trajectory in values:

        ax.plot(np.imag(trajectory[:, 0]), np.real(trajectory[:, 0]), np.real(trajectory[:, 1]))
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



trajs = compute_traj(trajdata)
#trajs = reconstitute(trajdata)

show_phasespace(N,dt, trajs)
#show_components(N,dt, trajs)
