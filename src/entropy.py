import numpy as np
import matplotlib.pyplot as plt
import plot_utils
from scipy.linalg import logm, expm
plt.rcParams.update(plt.rcParamsDefault)
def log2m(a):
    return logm(a)/np.log(2)

lmd_ = np.linspace(0,2,100)
s_00 = np.array([1,0,0,0])
s_10 = np.array([0,1,0,0])
s_01 = np.array([0,0,1,0])
s_11 = np.array([0,0,0,1])
s_0 = np.array([1,0])
s_1 = np.array([0,1])
pauli_x = np.matrix([[0,1],[1,0]])
pauli_z = np.matrix([[1,0],[0,-1]])

eps_list = [0,2.5,6.5,7]
H0 = np.diag(eps_list)

H_x = 2.0
H_z = 3.0
H_I = H_x * np.kron(pauli_x, pauli_x) + H_z * np.kron(pauli_z, pauli_z)
S_a_list = []
S_b_list = []
tol = 1e-3
for i in range(len(lmd_)):        
    H_tot = H0 + lmd_[i]*H_I
    eig_vals, eig_vecs = np.linalg.eig(H_tot)
    permute = eig_vals.argsort()
    eig_vals = eig_vals[permute]
    eig_vecs = eig_vecs[:,permute]
    #print(eig_vecs[:,0])
    DM = np.outer(eig_vecs[:,0], eig_vecs[:,0])
    #print(DM)
    d = np.matrix([[1,0],[0,1]])
    v1 = [1,0]
    proj1 = np.kron(v1,d)
    x1 = proj1@DM@proj1.T
    v2 = [0,1]
    proj2 = np.kron(v2,d)
    x2 = proj2@DM@proj2.T
    #print(proj1)
    #print()
    #print(proj2)
    #print(x2)
    S_a = -np.trace(x1 @ log2m(x1+tol))
    S_b = -np.trace(x2 @ log2m(x2+tol))
    S_a_list.append(S_a)
    S_b_list.append(S_b)


plt.plot(lmd_, S_a_list, label ='S_a')
plt.plot(lmd_, S_b_list, label = 'S_b')
plt.xlabel('$\lambda$')
plt.ylabel('Entropy')
plt.legend()
plot_utils.save("Entropy.pdf")
plt.show()