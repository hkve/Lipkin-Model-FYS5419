import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import plot_utils

def solve():
    H0 = np.eye(2)
    E1 = 0; E2 = 4
    H0[0,0] = E1; H0[1,1] = E2


    H1 = np.eye(2)
    v11 = 3; v22 = -v11; v12 = 0.2; v21 = v12
    H1[0,0] = v11; H1[0,1] = v12; H1[1,0] = v21; H1[1,1] = v22 

    pauli_x = np.matrix([[0,1],[1,0]])
    pauli_z = np.matrix([[1,0],[0,-1]])
    c = (v11 + v22)/2
    omega_z = (v11-v22)/2
    omega_x = v12


    n = 100
    lambdas = np.linspace(0,1,n)
    Es = np.zeros((n,2))
    C1s, C2s  = np.zeros_like(Es), np.zeros_like(Es)
    for i, lmd_ in enumerate(lambdas):
        H_ = H0 + lmd_*H1
        eig_val, eig_vec = np.linalg.eigh(H_)
        Es[i,:] = eig_val
        C1s[i,:] = eig_vec[:,0]
        C2s[i,:] = eig_vec[:,1] 

    return lambdas, Es[:,0], Es[:,1], C1s[:,0]**2, C2s[:,0]**2

def plot(lambdas, E1, E2, C_abs_E1, C_abs_E2):
    fig, ax = plt.subplots()
    cmap = plot_utils.cmap_terrain
    ax.scatter(lambdas, E1, label=r"$E_0$", c=C_abs_E1, cmap=cmap)
    ax.scatter(lambdas, E2, label=r"$E_1$", c=C_abs_E2, cmap=cmap)
    
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(np.linspace(0,1,100))
    cbar = plt.colorbar(sm)
    cbar.set_label(r"$|C_0|^2$")
    ax.set(xlabel=r"$\lambda$", ylabel="E [magic]")
    ax.legend()
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(lambdas, C_abs_E1, label=r"$E_0$")
    # ax.plot(lambdas, C_abs_E2, label=r"$E_1$")
    # ax.set(xlabel=r"$\lambda$", ylabel="E [magic]")
    # ax.legend()
    # plt.show()

    
def main():
    lambdas, E1, E2, C_abs_E1, C_abs_E2 = solve()

    plot(lambdas, E1, E2, C_abs_E1, C_abs_E2)

if __name__ == "__main__":
    main()