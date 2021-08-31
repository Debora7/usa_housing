import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools

# Citim datele din fisier
# data = np.loadtxt(os.path.join('Data', 'data.txt'), delimiter=',')
x = np.loadtxt(os.path.join('Data', 'data1.txt'))
y = np.loadtxt(os.path.join('Data', 'data2.txt'))
m = y.size

theta = np.array([27, 3])
x = np.stack([np.ones(m), x], axis=1)
iteratii = 1500
alfa = 0.01

# Functia eroare
def eroare(theta,x):
    h = np.dot(x,theta)
    return h

# Functia cost
def cost(x, y, theta):
    J = 0
    o = np.ones(m)
    squared_error = (eroare(theta, x) - y) ** 2
    J = np.dot(o, squared_error) / (2 * m)
    return J

# Functia gradient descendent
def grad(x, y, theta, alfa, nr_iteratii):
    theta = theta.copy()
    J_vechi = []
    theta_vechi = []

    for i in range(nr_iteratii):
        theta_vechi.append(list(theta))
        h = eroare(theta, x)
        theta[0] = theta[0] - (alfa / m) * (np.sum(h - y))
        theta[1] = theta[1] - (alfa / m) * (np.sum((h - y) * x[:, 1]))
        J_vechi.append(cost(x, y, theta))

    return theta, J_vechi, theta_vechi

theta = np.zeros(2)

theta, J_history, theta_history = grad(x ,y, theta, alfa, iteratii)
print('Parametrii theta obtinuti cu gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Parametrii theta asteptati (aproximativ): [-3.6361, 1.1670]')

plt.plot(x, y, 'rx')
plt.grid()
plt.plot(x[:, 1], eroare(theta, x), '-')
plt.legend(['Training data', 'Linear regression' + ' h(x) = %0.2f + %0.2fx'%(theta[0],theta[1])]);
plt.show()

# Realizarea predictiei
predict1 = eroare(theta, np.array([1, 3.5]))
print('Pentru populatia = 35,000, modelul prezice un profit de {:.2f}\n'.format(predict1*10000))

predict2 = eroare(theta, np.array([1, 7]))
print('Pentru populatia = 70,000, modelul prezice un profit de {:.2f}\n'.format(predict2*10000))

# Afisarea functiei eroare
def afis_fc_eroare(J_vechi):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(J_vechi)), J_vechi, 'bo')
    plt.grid(True)
    plt.title("Convergenta functiei eroare")
    plt.xlabel("Numarul de iteratii")
    plt.ylabel("Functia eroare")
    plt.show()

afis_fc_eroare(J_history)

# Afisarea lui J(theta)
# definirea figurii
fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

# grid de valori pentru care vom calcula costul J
theta0_vals = np.linspace(-10, 10, 50)
theta1_vals = np.linspace(-1, 4, 50)

# definim un vector J_vals care in care vom stoca valoarea costurilor
J_vals = []

# definim 2 vectori in care sa stocam valorile parametriilor
# theta folositi pentru calculul costlului J_vals
theta_0 = []
theta_1 = []

# calculam costul J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals.append(cost(x, y, [theta0, theta1]))
        theta_0.append(theta0)
        theta_1.append(theta1)

print(np.max(J_vals))
print(np.min(J_vals))
ax.scatter(theta_0,theta_1,J_vals,c=np.abs(J_vals),cmap=plt.get_cmap('YlOrRd'))

plt.xlabel(r'$\theta_0$',fontsize=20)
plt.ylabel(r'$\theta_1$',fontsize=20)
plt.title('Cost (Traiectoria parcursa pentru minimizare)',fontsize=16)
plt.plot([x[0] for x in theta_history],[x[1] for x in theta_history],J_history,'bo-')
plt.show()
