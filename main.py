import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cmath

#matplotlib.rcParams["text.usetex"] = True

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


class parameter_family:
    def __init__(self,epsilon,w1,w2,s):
        self.epsilon = epsilon
        self.w1 = w1
        self.w2 = w2
        self.s = s



def rectangular(ax, data, label):
    ax.plot(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]), label = label, color ='#576850' )
    return ax

def rectangular2(ax, data, label):
    ax.plot(np.imag(data[:, 1]), np.real(data[:, 1]), np.real(data[:, 0]), label = label, color ='#576850')

#colors = ['#576850',"#974E49",'#022B3A']

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
   # print(array)
    #print(np.round(array,5))
    #print(np.real(np.round(array,5)[1]))
    return array
def compute_trajectory_data(trajectory_skeleton, N, dt):
    A10 = trajectory_skeleton.initial_conditions[0]
    A20 = trajectory_skeleton.initial_conditions[1]
    print(trajectory_skeleton.label())

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
    print(label)
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


def stationary_points(A_1, s, w1, w2, e, q):

    A = np.abs(A_1)

    a = (s[1,1]-s[0,1])*A
    b = -e
    c = (w2-w1)*A + (s[1,0]-s[0,0])*A**3
    d = -e*q*A**2
    positive_roots = []


    for root in np.roots([a,b,c,d]):
        if np.real(root) >0:
            if np.abs(np.imag(root)) < 0.000001:
                positive_roots.append(root)


    return positive_roots
def stability_matrix(a, d, e, s, w1, w2):
    #print(-w1-s[0,0]*a**2-s[0,1]*d**2)


    r1 = np.array([0,-w1-s[0,0]*a**2-s[0,1]*d**2,e,0])
    r2 = np.array([w1+s[0,0]*3*a**2 + s[0,1]*d**2,0,0,e+2*s[0,1]*a*d])
   # r3 = np.array([e - 2 * s[1, 0] * d * a, 0, 0, -e*(a/d) - 2*s[1,1]*d**2 ])
    #r4 = np.array([0, e, e*a/d, 0])
    r3 = np.array([e-2*s[1,0]*d*a,0,0,-w2-s[1,0]*a**2-s[1,1]*3*d**2])
    r4 = np.array([0,e,w2+s[1,0]*a**2+s[1,1]*d**2,0])


    eigenvalues, eigenvectors =  np.linalg.eig(np.array([r1,r2,r3,r4]))


    return eigenvalues


def declare_stability(a, d, e, s, w1, w2):


    eigenvalues = stability_matrix(a, d, e, s, w1, w2)
    stable = True

    print(eigenvalues)
    for eigenvalue in eigenvalues:
        if np.abs(np.real(eigenvalue)) >0.00001:
            stable = False
    print(stable)
    return stable


def declare_stationary_stability(a, d, e, s, w1, w2):

    r = w1 + s[0,0]*a**2 + s[0,1]*d**2
    k = w2 + s[1,0]*a**2 + s[1,1]*d**2

    w = (r+k)/2 - np.sqrt(((r-k)/2)**2-e**2)
    A1 = a
    A2 = d
    exactness1 = (w1 - w) * A1 + e * A2 + A1 * (s[0, 0] * A1 ** 2 + s[0, 1] * A2 ** 2)
    exactness2 = -(w2 - w) * A2 + e * A1 - A2 * (s[1, 0] * A1 ** 2 + s[1, 1] * A2 ** 2)
    print("Exactness")
    print(exactness1, exactness2)



    return declare_stability(a,d,e,s,w1-w,w2-w)


def declare_alternate_stationary_stability(a, d, e, s, w1, w2):

    r = w1 + s[0,0]*a**2 + s[0,1]*d**2
    k = w2 + s[1,0]*a**2 + s[1,1]*d**2

    w = (r+k)/2  + np.sqrt(((r-k)/2)**2-e**2)
    A1 = a
    A2 = d
    exactness1 = (w1-w) * A1 + e * A2 + A1 * (s[0, 0] * A1 ** 2 + s[0, 1] * A2 ** 2)
    exactness2 = -(w2-w) * A2 + e * A1 - A2 * (s[1, 0] * A1 ** 2 + s[1, 1] * A2 ** 2)
    print("AlternateExactness")
    print(exactness1, exactness2)


    return declare_stability(a,d,e,s,w1-w,w2-w)

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

    A1s = np.roots([a,b,c1+c2,d1+d2,e1+e2])
    real_positive_A1s = []
    points = []

    for A1 in A1s:
        if np.imag(A1) ==0:
            if np.real(A1) > 0:
                real_positive_A1s.append(np.sqrt(np.real(A1)))

    for A1 in real_positive_A1s:
        a = A1 * r[0, 1]
        b = e
        c = w1 * A1 + A1 ** 3 * r[0, 0]
        A2s = np.roots([a, b, c])
        for A2 in A2s:
            if np.imag(A2) == 0:
                if np.real(A2) > 0:
                    points.append([A1,A2])
    for point in points:
        print("Stability of point ")
        print(point)
        declare_stability(point[0],point[1],e,r,w1,w2)
    return points

#declare_stationary_stability(1,1,0.1,s,2,3)


def generic_fixed_points():

    found_point = False
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
        points = fixed_points(s, w1, w2, epsilon)
        print("TESTING POINTS")
        tight_points = []
        for point in points:
            A1 = point[0]
            A2 = point[1]
            c = A1*w1+epsilon*A2+A1*(s[0,0]*A1**2 + s[0,1]*A2**2)
            d = -A2 * w2 + epsilon * A1 - A2 * (s[1, 0] * A1 ** 2 + s[1, 1] * A2 ** 2)
            if np.abs(c) <0.00001:
                if np.abs(d) <0.00001:
                    tight_points.append(point)
        if len(tight_points) >0:
            found_point = True
            print("found")
            print(tight_points)
            for point in tight_points:

                A1 = point[0]
                A2 = point[1]
                print("STABILITY of ")
                print(point)
                declare_stability(A1,A2,epsilon,s,w1,w2)
                print("asdkf")
                print(w1 * A1 + epsilon * A2 + A1 * (s[0, 0] * A1 ** 2 + s[0, 1] * A2 ** 2))
                print(w2 * A2 - epsilon * A1 + A2 * (s[1, 0] * A1 ** 2 + s[1, 1] * A2 ** 2))


        else:
            print("No Fixed Points Found in this System")
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel("$\mathrm{Re}(A_1)$", labelpad=8, fontsize="x-large")
    ax.set_ylabel("$\mathrm{Re}(A_2)$", labelpad=8, fontsize="x-large")
    ax.set_zlabel("$\mathrm{Im}(A_2)$", labelpad=14, fontsize="x-large")

  #  for point in points:
     #   ax = plt.figure().add_subplot(projection='3d')
     #   ax.set_xlabel("$\mathrm{Re}(A_1)$", labelpad=8, fontsize="x-large")
     #   ax.set_ylabel("$\mathrm{Re}(A_2)$", labelpad=8, fontsize="x-large")
     #   ax.set_zlabel("$\mathrm{Im}(A_2)$", labelpad=14, fontsize="x-large")
        #print("COMPUTING STABILITY FOR")
      #  print(point)
      #  stability_matrix(point[0], point[1], epsilon, s, w1, w2)
        #print(declare_stability(point[0], point[1], epsilon, s, w1, w2))
     #   if declare_stability(point[0], point[1], epsilon, s, w1, w2) == True:
    #    t = procure(epsilon, [point[0] + 0.000001, point[1]*1j + 0.00001],w1,w2,1,s,20000,0.0005)
    #    ax.scatter3D(point[0], 0, point[1], marker=(5, 2), s=100, color="red")
     #   ax.scatter3D(point[0] + 0.000001, 0.00001, point[1], marker=(5, 2), s=100, color="red")
    #    data = t.calculated_trajectory[2]
      #  print("data")
      #  print(data)
     #   print("data :,1")
     #   print(data[:, 1])
    #    ax.plot(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]))

    colors = ['#576850', "#974E49", '#022B3A',"#C4AF9A","#67523C","#6D6C37",'#576850', "#974E49", '#022B3A',"#C4AF9A","#67523C","#6D6C37"]
    for point in range(0, len(tight_points)):
        print(point)
        rotations1 = []
        rotations2 = []
        for i in np.linspace(0, 7, 40):
            rotations1.append(points[point][0] * cmath.exp(1j * i))
            rotations2.append(points[point][1] * cmath.exp(1j * i))
        ax.plot3D(np.real(rotations1), np.real(rotations2), np.imag(rotations2), colors[point])

    dt = 0.000005
    N = 1000
    A1 = points[0][0]
    A2 = points[0][1]
    A2 = A2*1j

#    t = procure(epsilon, [A1, A2], w1, w2, 1, s, N, dt)
 #   data = t.calculated_trajectory[2]
#    print(data)
 #   x = np.linspace(0, 1, len(data[:, 0]))
 #   ax2.plot(x, data[:, 0])

    plt.show()

#generic_fixed_points()


#generic_fixed_points()
def generic_stationary_points():
    ax = plt.figure().add_subplot(projection='3d')
    w1 = np.random.uniform(1,5)
    w2 = np.random.uniform(1,5)
    epsilon = np.random.uniform(0.1,0.5)
    q = 1
    s = np.ones((2, 2))
    s[0, 0] = np.random.uniform(-3,3)
    s[0, 1] = np.random.uniform(-3,3)
    s[1, 0] = np.random.uniform(-3,3)
    s[1, 1] = np.random.uniform(-3,3)


    print(epsilon)
    print(w1)
    print(w2)
    print(s)



    ax.set_xlabel("$\mathrm{Re}(A_1)$",labelpad = 8, fontsize = "x-large")
    ax.set_ylabel("$\mathrm{Re}(A_2)$",labelpad = 8, fontsize = "x-large")
    ax.set_zlabel("$\mathrm{Im}(A_2)$",labelpad = 14, fontsize = "x-large")
    points = []

    colors = ['#576850',"#974E49",'#022B3A']
    shuffled_colors = []

    for A1 in np.linspace(0.5, 10, 5):

        print("A1")
        print(A1)

        A2s = stationary_points(A1, s, w1, w2, epsilon, q)
        print(A2s)
        i = 0
        for A2 in A2s:
            points.append([A1,A2])
            shuffled_colors.append(colors[i])
            i = i+1
    print(points)
    print("found ")
    print(len(points))

    for point in range(0,len(points)):

        rotations1 = []
        rotations2 = []
        for i in np.linspace(0,7,40):
            rotations1.append(points[point][0]*cmath.exp(1j*i))
            rotations2.append(points[point][1]*cmath.exp(1j*i))
        ax.plot3D(np.real(rotations1), np.real(rotations2), np.imag(rotations2), color = shuffled_colors[point])

    plt.savefig("stationarypoints")
    plt.show()




def plot_number_of_fixed_points():
    s = np.array([[-1.04424256, -0.52166707],[ 1.87889378, -0.91559633]])
    ywidth = 5
    xwidth = 5
    fineness = 3
    epsilon = 0.3175533258995343

    x = np.linspace(0, xwidth, xwidth * fineness)
    y = np.linspace(0, ywidth, ywidth * fineness)
    Z = np.zeros([xwidth*fineness,ywidth*fineness])
    X, Y = np.meshgrid(x, y)

    for i in range(0, ywidth * fineness):
        for j in range(0,xwidth * fineness):
            Z[i][j] = len(fixed_points(s, i, 1, j))

    cmap = matplotlib.colors.LinearSegmentedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
    plt.pcolormesh(X, Y, Z,cmap)
    ax = plt.subplot()
    ax.set_xlabel('$\epsilon$')
    ax.set_ylabel('$w_1$')
    plt.colorbar(label="Number of Fixed Points")
    plt.show


def plot_stability_of_fixed_points():
    ywidth  = 10
    xwidth= 10
    fineness = 15
    s = np.array([[-0.36027231, -0.11098731],[ 0.97322085, -1.94585721]])
    epsilon = 0.24402154652862718
    x = np.linspace(0, xwidth, xwidth*fineness)
    y = np.linspace(0, ywidth, ywidth*fineness)
    Zstable = np.zeros([xwidth*fineness,ywidth*fineness])
    Zunstable = np.zeros([xwidth*fineness,ywidth*fineness])
    Z = np.zeros([xwidth * fineness, ywidth * fineness])
    X, Y = np.meshgrid(x, y)


    for county, i in enumerate(np.linspace(0, ywidth, ywidth*fineness)):
        for countx, j in enumerate(np.linspace(0, ywidth, ywidth*fineness)):

            points = fixed_points(s, i, 1, j)
            print("evaluating at ")
            print(i,j)
            Z[county][countx] = len(points)
            num_stable = 0
            num_unstable = 0
            for point in points:
                if declare_stability(point[0],point[1],j,s,i,1) ==True:
                    num_stable  = num_stable + 1
                else:
                    num_unstable = num_unstable+1
            Zstable[county][countx] = num_stable
            Zunstable[county][countx] =num_unstable

  #  ax = plt.subplot()
 #   ax2 = plt.subplot()
 #   plt.pcolormesh(X, Y, Zstable, cmap="Greys")
  #  ax2.pcolormesh(X, Y, Zunstable)

 #   ax.set_xlabel('$\\epsilon$')
#    ax.set_ylabel('$w_1$')
#    plt.colorbar(label="Number of Stable Fixed Points")
#    plt.show
   # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,8))
    ax1 = plt.figure(figsize = (5,5)).add_subplot()
    ax2 = plt.figure(figsize = (5,5)).add_subplot()
    ax3 = plt.figure(figsize = (5,6)).add_subplot()
    ax4 = plt.figure(figsize=(5, 6)).add_subplot()
    axes = [ax1,ax2,ax3]
    colors = ["#DADCD9","#B8CA96","#70A99A","#3C829C","#193A7F", "#000944"] #cmap=matplotlib.colors.ListedColormap(colors)

    im1 = axes[0].contourf(X,Y,Zstable,cmap="YlGnBu",levels = [0,1,2,3,4,5,6])
    im2 =axes[1].contourf(X,Y,Zunstable,cmap="YlGnBu",levels = [0,1,2,3,4,5,6])
    im = axes[2].contourf(X, Y, Z, cmap="YlGnBu", levels=[0, 1, 2, 3, 4, 5, 6])
    ax4.contourf(X, Y, np.add(Zstable,Zunstable), cmap="YlGnBu", levels=[0, 1, 2, 3, 4, 5, 6])
    axes[0].set_title("# of Stable Fixed Points",fontsize="xx-large")
    axes[1].set_title("# of Unstable Fixed Points", fontsize="xx-large")
    axes[2].set_title("# of Fixed Points", fontsize="xx-large")
    for i in range(0,3):
        axes[i].set_xlabel("$\\omega_1$",fontsize="xx-large")
        axes[i].set_ylabel("$\epsilon$",fontsize="xx-large")
        axes[i].set_yticks([0,1,2,3,4,5,6,7,8,9,10])

    plt.colorbar(im,orientation="horizontal") #ax=axes.ravel().tolist(),shrink=0.5)
    plt.tight_layout()

    plt.show()


def plot_stable_fixed_point():
    w1 = 2.1064011559361946    #312576282777
    w2 = 4.797283285862001     #99843618552925
    #  epsilon = np.random.uniform(0.1, 0.5)
    epsilon = 0.1610854945130396
    q = 1
    s = np.array([[ 2.684162  , -2.96716357],[-1.15710694 ,-0.66321078]])
    print("Tried ")
    print(s)
    print(w1, w2, epsilon)
    points = fixed_points(s, w1, w2, epsilon)
    colors = ['#576850', "#974E49", '#022B3A']


    for point in [points[2]]:
        ax = plt.figure(figsize = (8,8)).add_subplot(projection='3d')
        ax.set_xlabel("$\mathrm{Re}(A_0)$", labelpad=16, fontsize="xx-large")
        ax.set_ylabel("$\mathrm{Im}(A_0)$", labelpad=16, fontsize="xx-large")
        ax.set_zlabel("$\mathrm{Im}(A_1)$", labelpad=16,fontsize="xx-large")

        A1 = point[0]
        A2 = point[1]
        print("STABILITY of ")
        print(point)
        declare_stability(A1, A2, epsilon, s, w1, w2)
        print("asdkf")
        print(w1 * A1 + epsilon * A2 + A1 * (s[0, 0] * A1 ** 2 + s[0, 1] * A2 ** 2))
        print(w2 * A2 - epsilon * A1 + A2 * (s[1, 0] * A1 ** 2 + s[1, 1] * A2 ** 2))
        N = 2000

        t = procure(epsilon, [point[0] + 0.0001+0.001*1j, point[1] * 1j], w1, w2, 1, s, N, 0.005)
       # ax2 = plt.figure().add_subplot()

        data = t.calculated_trajectory[2]
      #  print(data)
      #  ax2.plot(np.linspace(0,1,len(np.real(data[:,1]))),np.real(data[:,1]))
        ax.plot(np.real(data[:, 0]), np.imag(data[:,0]), np.imag(data[:, 1]),color = colors[0])
        ax.scatter3D(point[0], 0, point[1], marker="D", s=600, color="black")
        #ax.scatter3D(np.real(data[N-1, 0]), np.imag(data[N-1,0]), np.imag(data[N-1, 1]), marker=(5, 2), s=200, color="red")

      #  ax.plot([np.real(data[N-1, 0]),np.real(data[N-1, 0])],[np.imag(data[N-1,0]),np.imag(data[N-1,0])-0.75],[np.imag(data[N-1, 1]),np.imag(data[N-1, 1])+1])
      #  ax.plot([np.real(data[N - 1, 0]), np.real(data[N - 1, 0])], [np.imag(data[N - 1, 0]), np.imag(data[N - 1, 0])-0.75],
           #     [np.imag(data[N - 1, 1]), np.imag(data[N - 1, 1]) - 1])



      #  ax.text3D(point[0], 0,1.6,"Fixed Point", backgroundcolor='white')
       # ax.legend(["*:Fixed Point"])
        ax.set_title("◆ = Fixed Point", fontsize="xx-large")
        plt.tight_layout()

       # ax.text(10, 0, 1,"* = Fixed Point", fontsize=14)


      #  ax.scatter3D(np.real(point[0]), np.real(point[1]), np.imag(point[1]), marker=(5, 2), s=100, color="red")







def plot_stationary_points():
    epsilon = 0.10772344954403504
    w1 = 2.440893031550415
    w2 = 2.8517955863319555
    s = np.array([[-2.57518181, - 2.50904539],[1.42017925, - 1.77898158]])
    ax = plt.figure().add_subplot(projection='3d')

    ax.set_xlabel("$\mathrm{Re}(A_0)$",labelpad = 8, fontsize = "x-large")
    ax.set_ylabel("$\mathrm{Re}(A_1)$",labelpad = 8, fontsize = "x-large")
    ax.set_zlabel("$\mathrm{Im}(A_1)$",labelpad = 14, fontsize = "x-large")
    points = []

    colors = ["#639e16","#974E49",'#022B3A','#576850']
    shuffled_colors = []

    for A1 in np.linspace(0.5, 10, 4):

        print("A1")
        print(A1)

        A2s = stationary_points(A1, s, w1, w2, epsilon, 1)
        print(A2s)
        i = 0
        for A2 in A2s:
            points.append([A1,A2])
            shuffled_colors.append(colors[i])
            i = i+1
    print(points)
    print("found ")
    print(len(points))

    for point in range(0,len(points)):

        rotations1 = []
        rotations2 = []
        for i in np.linspace(0,7,40):
            rotations1.append(points[point][0]*cmath.exp(1j*i))
            rotations2.append(1j*points[point][1]*cmath.exp(1j*i))
        ax.plot3D(np.real(rotations1), np.real(rotations2), np.imag(rotations2), color = colors[point],linewidth = 3)
  #  A1 = points[1][0]
  #  A2 = points[1][1]
  #  t = procure(epsilon,[A1,A2*1j],w1,w2,1,s,1000,0.001)
  #  data = t.calculated_trajectory[2]
  #  ax.plot3D(np.real(data[:,0]), np.real(data[:,1]), np.imag(data[:,1]), color="red")
    plt.title("Limit Cycles",fontsize="xx-large")
    plt.savefig("stationarypoints")
    plt.show()


def plot_fixed_points():
    w1 = 2.499764647448312
    w2 = 4.7789692235018
    epsilon = 0.18935052527387308

    q = 1
    s = np.array([[-1.44161228 ,-0.72719458],[-1.90048161 ,-2.68167117]])
    print("Tried ")

    points = fixed_points(s, w1, w2, epsilon)


    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel("$\mathrm{Re}(A_0)$", labelpad=8, fontsize="xx-large")
    ax.set_ylabel("$\mathrm{Re}(A_1)$", labelpad=8, fontsize="xx-large")
    ax.set_zlabel("$\mathrm{Im}(A_1)$", labelpad=14, fontsize="xx-large")

    colors = ['#576850', "#974E49", '#022B3A', "#C4AF9A", "#67523C", "#6D6C37", '#576850', "#974E49", '#022B3A', "#C4AF9A",
              "#67523C", "#6D6C37"]
    for point in range(0, len(points)):

        rotations1 = []
        rotations2 = []
        for i in np.linspace(0, 7, 40):
            rotations1.append(points[point][0] * cmath.exp(1j * i))
            rotations2.append(points[point][1] * cmath.exp(1j * i))
        ax.plot3D(np.real(rotations1), np.real(rotations2), np.imag(rotations2), colors[point],linewidth=3)

    plt.title("Fixed Points",fontsize="xx-large")
    plt.show()

def plot_generic():
    epsilon = 0.1
    w1 = 1
    w2 = 3
    s = np.array([[-2.57518181, 2], [1, - 1.77898158]])

    A1 = 1
    A2 = 3
    t = procure(epsilon, [A1, A2 * 1j], w1, w2, 1, s, 3000, 0.005)
    data = t.calculated_trajectory[2]
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]), color='#576850')
    ax.set_xlabel("$\mathrm{Re}(A_0)$", labelpad=16, fontsize="x-large")
    ax.set_ylabel("$\mathrm{Im}(A_0)$", labelpad=16, fontsize="x-large")
    ax.set_zlabel("$\mathrm{Im}(A_1)$", labelpad=16, fontsize="x-large")
    ax.set_title("Generic Solution")
    plt.show()
def plot_stable_limit_cycle():

    epsilon = 0.10772344954403504
    w1 = 2.440893031550415
    w2 = 2.8517955863319555
    s = np.array([[-2.57518181, - 2.50904539],[1.42017925, - 1.77898158]])
    ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')

    ax.set_xlabel("$\mathrm{Re}(A_0)$",labelpad = 8, fontsize = "x-large")
    ax.set_ylabel("$\mathrm{Re}(A_1)$",labelpad = 8, fontsize = "x-large")
    ax.set_zlabel("$\mathrm{Im}(A_1)$",labelpad = 14, fontsize = "x-large")
    points = []

    colors = ['#576850',"#974E49",'#022B3A']
    shuffled_colors = []

    for A1 in [8]:  #8 is stable

        print("A1")
        print(A1)

        A2s = stationary_points(A1, s, w1, w2, epsilon, 1)
        print(A2s)
        i = 0
        for A2 in A2s:
            points.append([A1,A2])
            shuffled_colors.append(colors[i])
            i = i+1
    print(points)
    print("found ")
    print(len(points))

    for point in points:
        A1 = point[0]
        A2 = point[1]
        t = procure(epsilon,[A1,A2*1j],w1,w2,1,s,100,0.001)
        data = t.calculated_trajectory[2]
        ax.plot3D(np.real(data[:,0]), np.real(data[:,1]), np.imag(data[:,1]), color="black",linewidth = 3)
        t = procure(epsilon, [A1+0.1, A2 * 1j+0.001], w1, w2, 1, s, 900, 0.001)
        data = t.calculated_trajectory[2]
        ax.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]), color=(87/255,104/255,80/255,0.9))
        plt.legend(["Stable Limit Cycle","Perturbed Orbit"])
    plt.title("Stable Limit Cycle",fontsize="xx-large")
    plt.savefig("stationarypoints")
    plt.show()

def plot_unstable_limit_cycle():
    epsilon = 0.10772344954403504
    w1 = 5
    w2 = 1
    s = np.array([[4.57518181, -2.50904539], [1.42017925, 1.77898158]])
    ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')

    ax.set_xlabel("$\mathrm{Re}(A_0)$",labelpad = 8, fontsize = "x-large")
    ax.set_ylabel("$\mathrm{Re}(A_1)$",labelpad = 8, fontsize = "x-large")
    ax.set_zlabel("$\mathrm{Im}(A_1)$",labelpad = 14, fontsize = "x-large")
    points = []

    colors = ['#576850',"#974E49",'#022B3A']
    shuffled_colors = []

    for A1 in [3]:  #8 is stable

        print("A1")
        print(A1)

        A2s = stationary_points(A1, s, w1, w2, epsilon, 1)
        print(A2s)
        i = 0
        for A2 in A2s:
            points.append([A1,A2])
            shuffled_colors.append(colors[i])
            i = i+1
    print(points)
    print("found ")
    print(len(points))

    for point in points:
        A1 = point[0]
        A2 = point[1]
        t = procure(epsilon,[A1,A2*1j],w1,w2,1,s,900,0.001)
        data = t.calculated_trajectory[2]
        ax.plot3D(np.real(data[:,0]), np.real(data[:,1]), np.imag(data[:,1]), color="black",linewidth = 3)
        t = procure(epsilon, [A1+0.1, A2 * 1j+0.001], w1, w2, 1, s, 5000, 0.001)
        data = t.calculated_trajectory[2]
        ax.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]), color=(87/255,104/255,80/255,0.9))
        plt.legend(["Unstable Limit Cycle","Perturbed Orbit"])
    plt.title("Unstable Limit Cycle",fontsize="xx-large")
    plt.savefig("stationarypoints")
    plt.show()


#plot_stable_fixed_point()

def generic_stability_of_periodic_orbits():
    fig = plt.figure()


    for i in range(1,10):
        found_something = False
        ax = fig.add_subplot(3, 3, i, projection='3d')
        while found_something == False:

            w1 = np.random.uniform(1, 5)
            w2 = np.random.uniform(1, 5)
            epsilon = np.random.uniform(0.1, 0.5)
            q = 1
            s = np.ones((2, 2))
            s[0, 0] = np.random.uniform(-3, 3)
            s[0, 1] = np.random.uniform(-3, 3)
            s[1, 0] = np.random.uniform(-3, 3)
            s[1, 1] = np.random.uniform(-3, 3)
            print("PARAMETERS")
            print(w1,w2,epsilon)
            print(s)
          #  ax = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')

            points = []

            colors = ['#576850', "#974E49", '#022B3A']
            shuffled_colors = []

            for A1 in [5]:  # 8 is stable


                A2s = stationary_points(A1, s, w1, w2, epsilon, 1)
                j = 0
                for A2 in A2s:
                    points.append([A1, A2])
                    shuffled_colors.append(colors[j])
                    j = j + 1
            stabilities_match = False
            if len(points)>0:
                A1 = points[0][0]
                A2 = points[0][1]



                stable = declare_stationary_stability(A1, A2, epsilon, s, w1, w2)
                otherstable = declare_alternate_stationary_stability(A1, A2, epsilon, s, w1, w2)
                if stable == otherstable:
                    stabilities_match = True



            if len(points)>0:

                if stabilities_match == True:


                        found_something = True
                        A1 = points[0][0]
                        A2 = points[0][1]


                        if stable == otherstable:
                            print("Stabilities match")
                            print(i)
                        else:
                            print("Stabilities dont match")
                            print(i)
                        t = procure(epsilon, [A1, A2 * 1j], w1, w2, 1, s, 500, 0.001)
                        data = t.calculated_trajectory[2]
                        ax.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]), color="black", linewidth=3)
                        t = procure(epsilon, [A1 + 0.01, A2 * 1j + 0.01], w1, w2, 1, s, 10000, 0.001)
                        data = t.calculated_trajectory[2]
                        ax.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]),
                                  color=(87 / 255, 104 / 255, 80 / 255, 0.9))
                        fig.tight_layout(h_pad=2)
                  #      axes = plt.figure().add_subplot(projection="3d")
                  #      axes.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 0]),
                   #               color=(87 / 255, 104 / 255, 80 / 255, 0.9))
                        if stable:
                            ax.title.set_text("Stable")
                        else:
                            ax.title.set_text("Unstable")




def stability_of_periodic_orbits():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    w1 = 1.1459876974733874
    w2 = 3.704432712382921
    epsilon = 0.49647125702479566


    s = np.array([[1.17428505, 1.11120771],[1.1952253 , 1.71681619]])

  #  ax = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')

    points = []

    colors = ['#576850', "#974E49", '#022B3A']
    shuffled_colors = []

    for A1 in [5]:  # 8 is stable


        A2s = stationary_points(A1, s, w1, w2, epsilon, 1)
        print(A2s)
        i = 0
        for A2 in A2s:
            points.append([A1, A2])
            shuffled_colors.append(colors[i])
            i = i + 1
    print(points)
    print("found ")
    print(len(points))


    if len(points)>0:
        found_something = True
        A1 = points[0][0]
        A2 = points[0][1]
        print("Declaring stability")
        stable = declare_stationary_stability(A1, A2, epsilon, s, w1, w2)
        t = procure(epsilon, [A1, A2 * 1j], w1, w2, 1, s, 5000, 0.0001)
        data = t.calculated_trajectory[2]
        ax.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]), color="black", linewidth=3)
        print("DOING THIS")
        t = procure(epsilon, [A1 + 0.01+0.001*1j, A2 * 1j ], w1, w2, 1, s, 20000, 0.0001)
        data = t.calculated_trajectory[2]
        ax.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 1]),
                  color=(87 / 255, 104 / 255, 80 / 255, 0.9))


        axes = plt.figure().add_subplot(projection="3d")
        axes.plot3D(np.real(data[:, 0]), np.real(data[:, 1]), np.imag(data[:, 0]),
                  color=(87 / 255, 104 / 255, 80 / 255, 0.9))
        if stable:
            ax.title.set_text("Stable")
        else:
            ax.title.set_text("Unstable")




def rotate(t):
    w = 0.1
    cos = np.cos(w*t)
    sin = np.sin(w*t)
    return np.array([[1,0,0,0],[0,cos,-1*sin,0],[0,sin,cos,0],[0,0,1,0]])



def plot_modulated_orbit():

    ax = plt.figure().add_subplot(projection="3d")

    theta = np.linspace(0,   3*np.pi, 1000)

    w = 3
    s = 0.5
    A1 = 2
    A2 = 1j

    a = A1 * np.exp(1j * (w ) * theta)
    b = 0

    x = np.real(a)
    y = np.imag(a)
    z = np.real(b)
   # ax.plot(x, y, z)
    xround = []
    yround = []
    zround = []
    for t in np.linspace(0,   3*np.pi, 1000):
        a = A1 * np.exp(1j * (w) * t)
        b = 0
        cos = np.cos(0.1*t)
        sin = np.sin(0.1*t)
        x = np.real(a)
        y = np.imag(a)
        z = np.real(b)
        xp = x * cos + sin * z
        yp = y
        zp = -sin * x + cos * z
        xround.append(xp)
        yround.append(yp)
        zround.append(zp)

   # plt.plot(np.array(xround),np.array(yround),np.array(zround))
    xr = []
    yr = []
    zr = []
    lr = []

    xrr = []
    yrr = []
    zrr = []
    lrr = []

    xrrr = []
    yrrr = []
    zrrr = []
    lrrr = []
    xrrb = []
    yrrb = []
    zrrb = []
    lrrb = []

    p = np.abs(A1)
    q = np.abs(A2)

    M = (1/(p**2 - q**2))*np.array([[p,0,0,-q],[0,p,q,0],[0,q,p,0],[-q,0,0,p]])
    N = np.linalg.inv(M)
    print(M)

    for t in np.linspace(0, 3 * np.pi, 1000):
        d = 0.1
      #  xr.append(A1*np.cos(w*t))
     #   yr.append(A1*np.sin(w*t)*np.cos(d*t))
      #  zr.append(A1*np.sin(w*t)*np.sin(d*t))
        xr.append(A1*np.cos(w*t))
        yr.append(A1*np.sin(w*t))
        zr.append(-1*np.abs(A2)*np.sin(w*t))
        lr.append( np.abs(A2) * np.cos(w*t))
        vec = np.array([[A1*np.cos(w*t)],[A1*np.sin(w*t)],[-1*np.abs(A2)*np.sin(w*t)],[np.abs(A2) * np.cos(w*t)]])

        rotatedvector = np.dot(M,vec)
        xrr.append(rotatedvector[0][0])
        yrr.append(rotatedvector[1][0])
        zrr.append(rotatedvector[2][0])
        lrr.append(rotatedvector[3][0])

        rotatedrotatedvector = np.dot(rotate(3),rotatedvector)
        xrrr.append(rotatedrotatedvector[0][0])
        yrrr.append(rotatedrotatedvector[1][0])
        zrrr.append(rotatedrotatedvector[2][0])
        lrrr.append(rotatedrotatedvector[3][0])

        rotatedback = np.dot(N,rotatedrotatedvector)

        xrrb.append(rotatedback[0][0])
        yrrb.append(rotatedback[1][0])
        zrrb.append(rotatedback[2][0])
        lrrb.append(rotatedback[3][0])




      #  print(rotatedvector)






    xr = np.array(xr)
    yr = np.array(yr)
    zr = np.array(zr)
    lr = np.array(lr)
    plt.plot(np.array(xr),np.array(yr),np.array(zr))
    plt.plot(np.array(xrr), np.array(yrr), np.array(zrr))
    plt.plot(np.array(xrrr), np.array(yrrr), np.array(zrrr))

    plt.plot(np.array(xrrb), np.array(yrrb), np.array(zrrb))




#    xpp = np.real(a)
 #   ypp = np.imag(a)
#    zpp = np.imag(-b)



#    xp = np.real(A1* np.exp(1j * (s) * theta))
#    yp = np.imag(-A1 * np.exp(1j * ( s) * theta))
 #   zp = np.real(A2*np.exp(1j*s*theta))
 #   ax.scatter(np.real(A1),np.imag(A1),np.real(A2))
 #   ax.scatter(np.real(A1*1j), np.imag(A1*1j), np.real(1j*A2))
  #  ax.scatter(np.real(-1*A1 * 1j), np.imag(-1*A1 * 1j), np.real(-1*1j * A2))
    ax.set_xlabel("re(A1),x")
    ax.set_ylabel("im(A1),y")
    ax.set_zlabel("re(A2),z")







  #  xpp = np.real(np.sum(A1 * np.exp(1j * (w ) * theta),np.exp(1j * (s) * theta)))
  #  ypp = np.imag(np.sum(A1 * np.exp(1j * (w ) * theta),(-1) * np.exp(1j * ( s) * theta)))
   # zpp = np.real(np.sum(A2 * np.exp(1j * (w) * theta),np.exp(1j*s*theta)))



  #  print(xpp.size)


    cos = np.cos(0.2)
    sin = np.sin(0.2)

    xp = x*cos +sin*z
    yp = y
    zp = -sin*x+cos*z

   # ax.plot(xp, yp, zp)
 #   ax.plot(xpp, ypp, zpp)
#    a = np.array([x[0], x[0]])
 #   b = np.array([y[0], y[0] + dx])
 #   c = np.array([z[0], z[0] + dx])
  #  ax.plot(a, b, c)



#    perturbationa= 1j*w*0.1
#    perturbationb = -1*w*0.1
 #   a = A1+ perturbationa
#    b=A2-perturbationb

#    ax.scatter(np.real(a), np.imag(a), np.real(b),marker='<')

  #  theta = np.linspace(0, 3 * np.pi, 1000)

 #   setb = b * np.exp(1j * (w) * theta)

#    print(setb)
   # plt.plot(np.real(seta),np.imag(seta),np.real(setb))




    plt.show()

def plot_block(block,ax):
    ax.plot(block[0, :], block[1, :], block[2, :])


def plot_modulated_orbit2():

    ax = plt.figure().add_subplot(projection="3d")

    t = np.linspace(0,   3*np.pi, 1000)

    w = 7
    A1 = 2.2
    A2 = 1j
    p = np.abs(A1)
    q = np.abs(A2)


    M = (1/(np.sqrt(p**2 - q**2)))*np.array([[p,0,0,-q],[0,p,q,0],[0,q,p,0],[-q,0,0,p]])
    N = np.linalg.inv(M)


    ax.set_xlabel("re(A1),x")
    ax.set_ylabel("im(A1),y")
    ax.set_zlabel("re(A2),z")

    orbit = np.array([p * np.cos(w * t), p * np.sin(w * t), -1 * q * np.sin(w * t), q * np.cos(w * t)])
    flattened_orbit = np.dot(M, orbit)

   # plot_block(orbit, ax)
    #plot_block(flattened_orbit, ax)

    for t in np.linspace(0,30,10):

        rotated_flattened_orbit = np.dot(rotate(t), flattened_orbit)
       # plot_block(rotated_flattened_orbit,ax)
        rotated_origional = np.dot(N,rotated_flattened_orbit)
      #  plot_block(rotated_origional, ax)


    trajectoryx = []
    trajectoryy = []
    trajectoryz = []


    for t in np.linspace(0,2,1000):
        d = 0.05
        trajectoryx.append(p*np.cos(w*t))
        trajectoryy.append(np.sin(w*t)*(p*np.cos(d*t)-q*np.sin(d*t)))
        trajectoryz.append(np.sin(w*t)*(p*np.sin(d*t)-q*np.cos(d*t)))

    plt.plot(np.array(trajectoryx),np.array(trajectoryy),np.array(trajectoryz))
    plt.show()




#plot_modulated_orbit2()

a = 1.1
b = 2.3
c = 7.2
d = 1.2

e = 5.6
f = 2.1
g = 2.3
h = 1.2


i = 2.3
k = 1.3
l = 2.9
m = 2.1


#print(np.dot(np.array([[a,b],[c,d]]),np.array([[e],[f]])))
#print(np.dot(np.array([[a,b],[c,d]]),np.array([[g],[h]])))
#r = np.dot(np.array([[a,b],[c,d]]),np.array([[e,g],[f,h]]))
#print(r)

#print(r[0,:])
s = np.ones((2, 2))

t = procure(0.1,[1,1],1,1,1,s,1000,0.1)
 #   data = t.calculated_trajectory[2]
#    print(data)
 #   x = np.linspace(0, 1, len(data[:, 0]))
 #   ax2.plot(x, data[:, 0])

ax = plt.figure().add_subplot(projection='3d')


data = t.calculated_trajectory[2]
ax.plot3D(np.real(data[:,0]), np.real(data[:,1]), np.imag(data[:,1]), color="red")

while True:
    plt.pause(5)



