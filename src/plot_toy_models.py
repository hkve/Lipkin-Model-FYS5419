from plot_levelcrossing import solve
import plot_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from qiskit import Aer
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import AerPauliExpectation
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import PauliSumOp

def get_hamiltonian(lmbda, qbits=1):
    hamiltonian = 0
    if qbits == 1:
        E1 = 0; E2 = 4
        v11 = 3; v22 = -v11; v12 = 0.2; v21 = v12

        eps, omega = (E1+E2)/2, (E1-E2)/2
        c, omegaz = lmbda*(v11 + v22)/2, lmbda*(v11-v22)/2
        omegax = lmbda*v12

        hamiltonian = PauliSumOp.from_list([
            ("I", eps),
            ("Z", omega),
            ("I", c),
            ("Z", omegaz),
            ("X", omegax),
        ])
    else:
        Hx = 2.0*lmbda
        Hz = 3.0*lmbda
        eps = [0.0, 2.5, 6.5, 7.0]

        eps_p1 = (eps[0] + eps[1])/2
        eps_m1 = (eps[0] - eps[1])/2
        eps_p2 = (eps[2] + eps[3])/2
        eps_m2 = (eps[2] - eps[3])/2
        ap = (eps_p1 + eps_p2)/2
        am = (eps_p1 - eps_p2)/2
        bp = (eps_m1 + eps_m2)/2
        bm = (eps_m1 - eps_m2)/2

        hamiltonian = PauliSumOp.from_list([
            ("II", ap),
            ("ZI", am),
            ("IZ", bp),
            ("ZZ", bm),
            ("XX", Hx),
            ("ZZ", Hz)
        ])

    return hamiltonian

def solve_toy_models(lmbda, qbits):
    ansatz = RealAmplitudes(num_qubits=qbits, entanglement='linear', reps=1)

    
    hamiltonian = get_hamiltonian(lmbda, qbits=qbits)
    backend = Aer.get_backend('statevector_simulator')
    optimizer = SPSA(maxiter=2000)
    expectation = AerPauliExpectation()


    vqe = VQE(ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=backend,
            expectation=expectation,)
    
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    print(result.optimal_value, lmbda)
    return result.optimal_value

def solve2(n=1000):
    lambdas  = np.linspace(0,1, n)
    E = np.zeros_like(lambdas)

    pauli_x = np.matrix([[0,1],[1,0]])
    pauli_z = np.matrix([[1,0],[0,-1]])

    eps_list = [0,2.5,6.5,7]
    H0 = np.diag(eps_list)

    H_x = 2.0
    H_z = 3.0
    H_I = H_x * np.kron(pauli_x, pauli_x) + H_z * np.kron(pauli_z, pauli_z)
    for i, lmd_ in enumerate(lambdas):
        H_tot = H0 + lmd_*H_I
        eig_vals, eig_vecs = np.linalg.eigh(H_tot)
        E[i] = eig_vals[0]

    return lambdas, E

def plot_one_qbit():
    n_diag, n_vqe = 1000, 10
    lambdas_diag, E1, _, C, _ = solve(n_diag)
    lambdas_vqe = np.linspace(0, 1, n_vqe)
    E_vqe = np.zeros_like(lambdas_vqe)

    for i, l in enumerate(lambdas_vqe):
        E_vqe[i] = solve_toy_models(lambdas_vqe[i], qbits=1)

    fig, ax = plt.subplots()
    ax.plot(lambdas_diag, E1, label=r"Diagonalization")
    ax.scatter(lambdas_vqe, E_vqe, marker="+", color="r", label="VQE")

    ax.legend()
    ax.set(xlabel=r"$\lambda$", ylabel=r"$E_0$")
    plot_utils.save("toy1")
    plt.show()

def plot_two_qbit():
    n_diag, n_vqe = 1000, 10
    lambdas_diag, E1 = solve2(n_diag)
    lambdas_vqe = np.linspace(0, 1, n_vqe)
    E_vqe = np.zeros_like(lambdas_vqe)

    for i, l in enumerate(lambdas_vqe):
        E_vqe[i] = solve_toy_models(lambdas_vqe[i], qbits=2)

    fig, ax = plt.subplots()
    ax.plot(lambdas_diag, E1, label=r"Diagonalization")
    ax.scatter(lambdas_vqe, E_vqe, marker="+", color="r", label="VQE")

    ax.legend()
    ax.set(xlabel=r"$\lambda$", ylabel=r"$E_0$")
    plot_utils.save("toy2")
    plt.show()

if __name__ == "__main__":
    # plot_one_qbit()
    plot_two_qbit()