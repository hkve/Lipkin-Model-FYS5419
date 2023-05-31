from plot_levelcrossing import solve
import plot_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import AerPauliExpectation
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import PauliSumOp

def solve_one_qbit(lmbda):
    ansatz = RealAmplitudes(num_qubits=1, entanglement='linear', reps=1)
    
    E1 = 0; E2 = 4
    v11 = 3; v22 = -v11; v12 = 0.2; v21 = v12

    eps, omega = (E1+E2)/2, (E1-E2)/2
    c, omegaz = lmbda*(v11 + v22)/2, lmbda*(v11-v22)/2
    omegax = lmbda*v12

    print(eps, omega, c, omegaz, omegax)
    exit()
    # I_term = Ep + Vp
    # Z_term = Em + Em
    # X_term = Vo

    hamiltonian = PauliSumOp.from_list([
        ("I", eps),
        ("Z", omega),
        ("I", lmbda*c),
        ("Z", lmbda*omegaz),
        ("X", lmbda*omegax),
    ])

    def callback(*args):
        print(args)
    backend = Aer.get_backend('statevector_simulator')
    optimizer = SPSA(maxiter=1000, callback=callback)
    expectation = AerPauliExpectation()


    vqe = VQE(ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=backend,
            expectation=expectation,)
    
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
   
    return result.optimal_value


def plot_one_qbit():
    n_diag, n_vqe = 1000, 10
    lambdas_diag, E1, _, C, _ = solve(n_diag)
    lambdas_vqe = np.linspace(0, 1, n_vqe)
    E_vqe = np.zeros_like(lambdas_vqe)

    for i, l in enumerate(lambdas_vqe):
        E_vqe[i] = solve_one_qbit(lambdas_vqe[i])

    fig, ax = plt.subplots()
    ax.plot(lambdas_diag, E1, label=r"Diagonalization")
    ax.scatter(lambdas_vqe, E_vqe, marker="+", color="r", label="VQE")

    ax.legend()
    plt.show()

def solve_two_qbit():
    pass

def plot_two_qbit():
    pass

if __name__ == "__main__":
    plot_one_qbit()