import plot_utils
import lipkin_utils

import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import AerPauliExpectation
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA

        
def run_lipkin_VQE(v, w, N=4, maxiter=1000):
    ansatz = RealAmplitudes(num_qubits=N, entanglement='linear', reps=1)
    
    hamiltonian = lipkin_utils.get_hamiltonian(v, w, N)

    backend = Aer.get_backend('statevector_simulator')
    optimizer = SPSA(maxiter=maxiter)
    expectation = AerPauliExpectation()

    vqe = VQE(ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=backend,
            expectation=expectation)

    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    return  result.optimal_value

def run_lipkin_DIAG(v, w, N=4):
    H = lipkin_utils.get_hamiltonian_matrix(v, w, N)
    E, C = np.linalg.eigh(H)
    return E[0]

def run_HF(v, w, N=4):
    v_tilde = v*(N-1)
    
    cond = 1
    front = -N/2
    if v_tilde > 1:
        cond = (1+(N-1)**2*v**2) / (2*(N-1)*v)

    return front*cond

def run_RPA(v, w, N=4):
    v_tilde = v*(N-1)

    A, B = 1,-(N-1)*v

    if v_tilde > 1:
        A = 3*(N-1)**2 * v / (2*(N-1)*v - 1)
        B = - (N-1)**2 * v**2 / (2*(N-1)*v+1)
    
    omega = np.sqrt(A**2 - B**2)
    return run_HF(v, w, N) + (omega - A)/2

def plot_v_vs_E(N=4):
    n_pts = 21
    v = np.linspace(0,2, n_pts)
    w = 0
    E_DIAG = np.zeros_like(v)
    E_VQE = np.zeros_like(v)
    E_HF = np.zeros_like(v)
    E_RPA = np.zeros_like(v)


    for i in range(n_pts):
        E_DIAG[i] = run_lipkin_DIAG(v[i], w, N)
        E_VQE[i] = run_lipkin_VQE(v[i], w, N, maxiter=100)
        E_HF[i] = run_HF(v[i], w, N)
        E_RPA[i] = run_RPA(v[i], w, N)

    fig, ax = plt.subplots()
    ax.plot(v, E_DIAG, c="gray", label="Diagonalization")
    ax.scatter(v, E_VQE, marker="x", c="r", label="VQE")
    ax.plot(v, E_HF, marker="+", markersize=4, label="HF")
    ax.plot(v, E_RPA, label="RPA")
    ax.set(xlabel=r"$V/\epsilon$", ylabel=r"$E_0 / \epsilon$")
    ax.legend()
    plot_utils.save("N2_lipkin_vary_V")
    plt.show()

if __name__ == '__main__':
    plot_v_vs_E(N=2)