import numpy as np





def g(x):
    A1 = x[0] + 1j*x[1]
    A2 = x[2] + 1j*x[3]
  #  print("A1 A2")
 #   print(A1)
 #   print(A2)
    q = 1

    array = np.zeros(2, dtype=np.complex64)

    array[0] = 1j * w1 * A1 + eps * A2 + 1j * A1 * (s[0, 0] * np.abs(A1) ** 2 + s[0, 1] * np.abs(A2) ** 2)
    array[1] = 1j * w2 * A2 + eps * q * A1 + 1j * A2 * (s[1, 0] * np.abs(A1) ** 2 + s[1, 1] * np.abs(A2) ** 2)

    return np.array([np.real(array[0]), np.imag(array[0]),np.real(array[1]), np.imag(array[1])])


def F(z,x):
    return(g(np.add(x,z))-g(x))

def V(z):
    return np.array([z[0]**2, z[1]**2,z[2]**2,z[3]**2])

def gradientV(z):
    return np.array([z[0]*2, z[1]*2,z[2]*2,z[3]*2])


def Vprime(z,x):
    return np.dot(gradientV(z),F(z,x))

print("SLDKJFLWE")

print(A1)
print(A2)

#A1 = A1*np.exp(-1j*np.pi)
#A2 = A2*np.exp(-1j*np.pi)


print(A1)
print(A2)
x = [A1,0,0,np.imag(A2)]
zpoints = [np.linspace(0,10,10),np.linspace(0,10,10),np.linspace(0,10,10),np.linspace(0,10,10)]

print(zpoints)
print(zpoints[0])
z = [zpoints[0][1],zpoints[0][2],zpoints[0][3],zpoints[0][4]]


for i in range(0,10):
    for j in range(0,10):
        for k in range(0,10):
            for l in range(0,10):
                #print(zpoints[0][i],zpoints[0][k],zpoints[0][k],zpoints[0][l])
                z = [zpoints[0][i],zpoints[0][k],zpoints[0][k],zpoints[0][l]]
                #print(Vprime(z,x))
                if Vprime(z,x) >0:
                    print("found positive")
#ax = plt.figure().add_subplot(projection='3d')
#ax.plot3D()
print(x)
print(V(x))
print(Vprime(z,x))


#ax.plot3D(np.imag(data[:, 0]), np.real(data[:, 0]), np.real(data[:, 1]))
#plt.show()





