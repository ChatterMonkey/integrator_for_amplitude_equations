import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cmath
from os.path import exists

from matplotlib.patches import Rectangle

eps = 0.01
dt = 0.0005
N = 100
plt.ion()
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
        label = self.label()
        basis(ax,data, label)
        plt.legend()
        plt.savefig

def rectangular(ax, data, label):
    ax.plot(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]), label = label)

def rectangular2(ax, data, label):
    ax.plot(np.imag(data[:, 1]), np.real(data[:, 1]), np.real(data[:, 0]), label = label)



def flat(ax, data, label):
    #plt.scatter()

    ax.plot()
    print(data)
    ax.plot(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1])**2 + np.imag(data[:,1])**2, label = label)

def f(trajectory_skeleton, A):
    A1, A2 = A
    w1 = trajectory_skeleton.omega1
    w2 = trajectory_skeleton.omega2
    s = trajectory_skeleton.s
    q = trajectory_skeleton.q
    eps = trajectory_skeleton.epsilon

#    print('eps')
 #   print(eps)
  #  print(w2)
   # print(np.abs(A1) ** 2)
    #print(np.abs(A2) ** 2)
    #print(s[0, 0] * np.abs(A1) ** 2 + s[0, 1] * np.abs(A2) ** 2)
   # print(1j * w1 * A1 + eps * A2)
   # print(eps * A2)
   # print(1j * w1 * A1)
    array = np.zeros(2, dtype=np.complex64)

    array[0] = 1j * w1 * A1 + eps * A2 + 1j * A1 * (s[0, 0] * np.abs(A1) ** 2 + s[0, 1] * np.abs(A2) ** 2)
    array[1] = 1j * w2 * A2 + eps * q * A1 + 1j * A2 * (s[1, 0] * np.abs(A1) ** 2 + s[1, 1] * np.abs(A2) ** 2)
   # print("array")
  #  print(array)
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


def mags(epsilon, w1,w2):
    w = (w2 - w1)/2
    mag = (w + np.sqrt(w**2 - epsilon**2))/epsilon
    return mag





def find_stationary_points(A_1, s, w1, w2, e, q):

    A = np.abs(A_1)

    a = (s[1,1]-s[0,1])*A
    b = -e
    c = (w2-w1)*A + (s[1,0]-s[0,0])*A**3
    d = -e*q*A**2

    return np.roots([a,b,c,d])




def find_fixed_points(s,w1,w2,epsilon,q):
    S = np.linalg.det(s)
    W = w2*s[0,1] - w1*s[1,1]
    R = W/S
    w2s = w2/s[1,0]
    e = s[1,1]*epsilon/(s[1,0]*S**2)



    a = 1
    b = w2s - 2*R
    c = R**2 - 2*w2s*R + e**2*((s[0,0]*s[1,0]*s[1,1])/s[0,1] - q*(2*s[1,0]*s[0,1] - S)+s[1,0]**2)
    d = w2s*R**2   + e**2*((s[1,1]/s[0,1])*(w2*s[0,0]+w1*s[1,0])-q*(2*w2*s[0,1]+ W))
    #e = e**2*((w1*w2)/(s[1,1]*s[0,1]**3)+(q*e**2*S**2)/(s[1,1]**2*s[0,1]**2))
    h = (epsilon**2*w1*w2 + epsilon**4*q)/(s[0,1]**3)



    f = epsilon**2/(s[0,1]**2)
    g = epsilon**2/(s[0,1]*s[1,1])
    s2 = s[1,0]/s[1,1]
    s1 = s[0,0]/s[0,1]
    omega1 = w1/s[0,1]
    omega2 = w2/s[1,1]



    a2 = f*s2*(s2-s1)**2
    b2 = 2*f*s2*(s2-s1)*(omega2-omega1) + f*omega2*(s2-s1)**2
    c2 = f*s2*(omega2-omega1)**2 + 2*f*omega2*(s2-s1)*(omega2-omega1) + f**2*s1*s2 - g*f*(3*s2-s1) + g**2
    d2 = f*omega2*(omega2-omega1)**2 + f**2*(omega2*s1+omega1*s2) - g*f*(3*omega2-omega1)
    f2 = f**2*omega1*omega2 + g*f**2


    print("COEFFIENENTS")
    print(a)
    print(a2)
    print(b2)
    print(c2)
    print(d2)
    print(f2)

    roots = np.roots([a2,b2,c2,d2,f2])

    for root in roots:
        if (np.imag(root) ==0 )& (np.real(root) > 0):
            print("Valid root")
            print(root)
            alpha = s[1,1]
            beta = 0
            gamma = s[1,0]*np.real(root) + w2
            delta = -epsilon*q*np.sqrt(np.real(root))
            print(np.roots([alpha,beta,gamma,delta]))


            print(np.roots([np.sqrt(np.real(root))*s[0,1],epsilon,w1*np.sqrt(np.real(root))+s[0,0]*np.sqrt(np.real(root))**3]))



            print(root)

    return roots




def fixed_points(r, w1, w2, e):
#def fixed_points(w,o,s,t,e,r):

    w = (w1 * r[1, 1] - w2 * r[0, 1]) / 2
    o = (w1 * r[1, 1] + w2 * r[0, 1]) / 2
    s = (r[0, 0] * r[1, 1] - r[0, 1] * r[1, 0]) / 2
    t = (r[0, 0] * r[1, 1] + r[0, 1] * r[1, 0]) / 2

    a = (4 * (e**2) * (r[1,1] / r[0,1]) )* s**2*(t+s)
    b = (4 * (e**2) * (r[1,1] / r[0,1]) )*(2*w*s*t+s**2*o+3*w*s**2)
    c1 = (4 * (e**2) * (r[1,1] / r[0,1]) )*(3*w**2*s+w**2*t+2*s*w*o)
    c2 = e**4*((r[1,1]*r[0,1])**2 + (r[1,1]/r[0,1])**2*(t**2 - s**2)-2*(r[1,1])**2*(t+2*s))
    d1 = (4 * (e**2) * (r[1,1] / r[0,1]) )*(o+w)*w**2
    d2 = 2*e**4*r[1,1]**2*((1/r[0,1])**2*(o*t-w*s)-o-2*w)
    e1 = e**4*(r[1,1] / r[0,1])**2*(o**2-w**2)
    e2 = e**6*(r[1,1]**3/r[0,1])
    return np.roots([a,b,c1+c2,d1+d2,e1+e2])


#    print(a)
 #   print(b)
  #  print(c1+c2)
   # print(d1+d2)
#    print(e1+e2)
 #   print(np.roots([a,b,c1+c2,d1+d2,e1+e2]))













#print(A1)

#a = A1*s[0,1]
#b = epsilon
#c = w1*A1 + A1**3*s[0,0]



#print(np.roots([a,b,c]))

#print("SKJHF")

#print(w1*A1+epsilon*A2+A1*(s[0,0]*A1**2+s[0,1]*A2**2))
#print(w2*A2-epsilon*A1+A2*(s[1,0]*A1**2+s[1,1]*A2**2))




#print(find_fixed_points(s,w1,w2,epsilon,1))

def fixed_point_mag(A_1, s,w1,w2,epsilon):
    A = np.abs(A_1)
    a = -epsilon*s[1,1]
    b = (A*(s[0,1]*w2 - s[1,1]*w1) + A**3*(s[0,1]*s[1,0] - s[1,1]*s[0,0]))
    c = A**2*s[0,1]*epsilon

    print(np.roots([s[0,1],epsilon,w1 + s[0,0]]))
    print(np.roots([s[1,1],0,w2+s[1,0],epsilon]))

    print(a)
    print(b)
    print(c)

    roots = np.roots([a,b,c])

    print(roots)

    return roots[1]


def J(w1,w2,epsilon):

    return -(w1+w2)/(2) + np.sqrt(((w1-w2)/2)**2-epsilon**2)


def generic_stationary_points():
    ax = plt.figure().add_subplot(projection='3d')
    matplotlib.pyplot.xlabel
    w1 = np.random.uniform(1,5)
    w2 = np.random.uniform(1,5)
    epsilon = np.random.uniform(0.1,0.5)
    q = 1
    s = np.ones((2, 2))
    s[0, 0] = np.random.uniform(-3,3)
    s[0, 1] = np.random.uniform(-3,3)
    s[1, 0] = np.random.uniform(-3,3)
    s[1, 1] = np.random.uniform(-3,3)
    ax.set_xlabel("A1")
    ax.set_ylabel("Re(A2)")
    ax.set_zlabel("Im(A2)")
    points = []
    for A1 in range(1,5):
        A2s = find_stationary_points(A1, s, w1, w2, epsilon, q)
        for A2 in A2s:
            points.append([A1,A2])
    print(points)
    for point in range(0,len(points)):
        print("POint:")
        print(point)
        rotations1 = []
        rotations2 = []
        for i in np.linspace(0,7,40):
            rotations1.append(points[point][0]*cmath.exp(1j*i))
            rotations2.append(points[point][1]*cmath.exp(1j*i))
      #  ax.plot3D(np.real(rotations1), np.real(rotations2), np.imag(rotations2))

    t = procure(epsilon,[points[4][0],points[4][1]*1j],w1,w2,1,s,1000,0.01)
    data = t.calculated_trajectory[2]
    ax.plot3D(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]))
    t = procure(epsilon, [points[4][0]+0.05, (points[4][1]+0.05) * 1j], w1, w2, 1, s, 1000, 0.01)
    data = t.calculated_trajectory[2]
    ax.plot3D(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]))

    print(rotations1)
    plt.savefig("stationarypoints")
    plt.show()







def compute_stability(a,d, e,s,w1,w2):
    r1 = np.array([0,-w1-s[0,0]*a**2-s[0,1]*d**2,e,0])
    r2 = np.array([w1+s[0,0]*3*a**2 + s[0,1]*d**2,0,0,e+2*s[0,1]*a*d])
    r3 = np.array([e-2*s[1,0]*d*a,0,0,-w2-s[1,0]*a**2-s[1,1]*3*d**2])
    r4 = np.array([0,e,w2+s[1,0]*a**2+s[1,1]*d**2,0])
    print(np.array([r1,r2,r3,r4]))
    eigenvalues, eigenvectors =  np.linalg.eig(np.array([r1,r2,r3,r4]))
    print(eigenvalues)
    print(eigenvectors)

def generic_fixed_points():
    ax = plt.figure().add_subplot(projection='3d')
    ax2 = plt.figure().add_subplot()
    ax.set_xlabel("A1")
    ax.set_ylabel("Re(A2)")
    ax.set_zlabel("Im(A2)")
    matplotlib.pyplot.xlabel
    found_point = False
    points = []
    while found_point == False:
        w1 = np.random.uniform(1, 5)
        w2 = np.random.uniform(1, 5)
        epsilon = np.random.uniform(0.1, 0.5)
        q = 1
        s = np.ones((2, 2))
        s[0, 0] = np.random.uniform(-3, 3)
        s[0, 1] = np.random.uniform(-3, 3)
        s[1, 0] = np.random.uniform(-3, 3)
        s[1, 1] = np.random.uniform(-3, 3)
        print("Tried ")
        print(s)
        print(w1,w2,epsilon)

        A1s = fixed_points(s, w1, w2, epsilon)
        real_positive_A1s = []

        for A1 in A1s:
            if np.imag(A1) ==0:
                if np.real(A1) > 0:
                    real_positive_A1s.append(np.sqrt(np.real(A1)))

        for A1 in real_positive_A1s:
            a = A1 * s[0, 1]
            b = epsilon
            c = w1 * A1 + A1 ** 3 * s[0, 0]
            A2s = np.roots([a, b, c])
            for A2 in A2s:
                if np.imag(A2) == 0:
                    if np.real(A2) > 0:
                        points.append([A1,A2])
        if len(points) != 0:
            found_point = True
            print("found")
            print(points)
            for point in points:
                A1 = point[0]
                A2 = point[1]
                print(w1 * A1 + epsilon * A2 + A1 * (s[0, 0] * A1 ** 2 + s[0, 1] * A2 ** 2))
                print(w2 * A2 - epsilon * A1 + A2 * (s[1, 0] * A1 ** 2 + s[1, 1] * A2 ** 2))


        else:
            print("NO pOINTS found")

    for point in points:
        print("COMPUTING STABILITY FOR")
        print(point)
        compute_stability(point[0],point[1],epsilon,s,w1,w2)
    for point in range(0, len(points)):
        print(point)
        rotations1 = []
        rotations2 = []
        for i in np.linspace(0, 7, 40):
            rotations1.append(points[point][0] * cmath.exp(1j * i))
            rotations2.append(points[point][1] * cmath.exp(1j * i))
       # ax.plot3D(np.real(rotations1), np.real(rotations2), np.imag(rotations2))

    dt = 0.000005
    N = 1000
    A1 = points[0][0]
    A2 = points[0][1]
    A2 = A2*1j

    t = procure(epsilon, [A1, A2], w1, w2, 1, s, N, dt)
    data = t.calculated_trajectory[2]
    print(data)
    x = np.linspace(0, 1, len(data[:, 0]))
    ax2.plot(x, data[:, 0])
   # plt.show()

  #  print(rotations1)
   # plt.savefig("stationarypoints")
  #  plt.show()


#generic_fixed_points()

s = np.ones((2, 2))
s[0, 0] = 1
s[0, 1] = -1
s[1, 0] = 1
s[1, 1] = -1
w1 = -0.1
w2 = 0.1


#STABILITY



















#def generic_fixed_points():


    #ax.scatter3D(np.real(rotations1),np.real(rotations2),np.imag(rotations2))



#A2 = find_stationary_points(A1, s, w1, w2, epsilon, q)[0]



#generic_stationary_points()


#print("allowed J")

#print(J)

#print(s)



s = np.ones((2,2))
s[0,0] = 2.69794158
s[0,1] = -2.86653
s[1,0] = -1.6156136
s[1,1] = 0.58899522

s = np.array([[-1.70123932,  1.11197577],[-0.27384857 ,-1.50748744]])

print(s)
w1 =3.4619226549338715
w2 = 4.00550832602082

epsilon = 0.31569359343931824
N = 1000
dt = 0.0005
#fixed_points(s,w1,w2,epsilon)
#A1 = 0.91162341685558
#A2 = 0.030803383507949967*1j
A1 = 1.823289021950946
A2 = 1.3288469348758136*1j
A1 = A1 + 0.05
A2 = A2 + 0.05

#compute_stability(1.823289021950946,1.3288469348758136, epsilon,s,w1,w2)

generic_stationary_points()


#t = procure(epsilon, [A1,A2], w1,w2, 1, s, N, dt)
#ax = plt.axes()
#print(t.calculated_trajectory[2])

#t.plot(ax,N,dt,rectangular)


#data = t.calculated_trajectory[2]

#x = np.linspace(0, 1, len(data[:, 0]))
#ax.plot(x, data[:, 0]-1.823289021950946)
#ax.plot(x, data[:, 1]-1.823289021950946)



#display_together([t], rectangular2)



#print(N*dt)
#h = procure(0.1, [1,1], 1,1,1,s2, 10000, 0.0005)
#print(t.calculated_trajectory[2][:,0])


def omega(w1,w2,A1,A2,epsilon):
    J = np.abs(A1)**2 - np.abs(A2)**2
    omega = (w1+w2)/2 + J + np.sqrt(((w1-w2)**2)/4 - epsilon**2)
    print(np.sqrt(((w1-w2)**2)/4 - epsilon**2))
    print(omega)
    return omega

#print("omega")
#print(2*np.pi/omega(w1,w2,A1,A2,epsilon))
#plt.plot(np.arange(t.calculated_trajectory[0])*t.calculated_trajectory[1],np.real(t.calculated_trajectory[2][:,1]) )
#plt.show()
#display_together([t], rectangular2)

#t.serialize()

#plt.draw()

while True:
    plt.pause(5)
