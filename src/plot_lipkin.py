import plot_utils
import lipkin_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import AerPauliExpectation
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA

def run_lipkin_VQE(v, w, N=4, maxiter=1000, force_nq_scheme=None, initial_point=None):
    ansatz = RealAmplitudes(num_qubits=N, entanglement='linear', reps=1)
    
    hamiltonian = lipkin_utils.get_hamiltonian(v, w, N, force_nq_scheme)

    backend = Aer.get_backend('statevector_simulator')
    optimizer = SPSA(maxiter=maxiter)
    expectation = AerPauliExpectation()

    vqe = VQE(ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=backend,
            expectation=expectation,
            initial_point=initial_point)

    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    
    if initial_point:
        return  result.optimal_value, result.optimal_point
    else:
        return result.optimal_value
    
def run_lipkin_DIAG(v, w, N=4):
    H = lipkin_utils.get_hamiltonian_matrix(v, w, N)
    E, C = np.linalg.eigh(H)
    return E[0]

def run_HF(v, w, N=4):
    R = (v+w)*(N-1)
    
    cond = 1
    front = -N/2
    if R > 1:
        cond = (1+(N-1)**2*(v+w)**2) / (2*(N-1)*(v+w))

    return front*cond + w

def run_RPA(v, w, N=4):
    R = (v+w)*(N-1)

    A, B = 1-(N-1)*w,-(N-1)*v

    if R > 1:
        A = (3*(N-1)**2 * (v+w)**2 - 1) / (2*(N-1)*(v+w)) - (N-1)*w
        B = - (1 + (N-1)**2 * (v+w)**2 )/ (2*(N-1)*(v+w)) + (N-1)*w
    
    omega = np.sqrt(A**2 - np.abs(B)**2)
    return run_HF(v, w, N) + (omega - A)/2

def plot_v_vs_E(N=4):
    n_pts = 21
    v = np.linspace(0,2, n_pts)
    w = 0
    E_DIAG = np.zeros_like(v)
    E_VQE = np.zeros_like(v)
    E_VQE_nq = np.zeros_like(v)
    E_HF = np.zeros_like(v)
    E_RPA = np.zeros_like(v)

    pt_nq = None
    for i in range(n_pts):
        E_DIAG[i] = run_lipkin_DIAG(v[i], w, N)
        E_VQE[i], _ = run_lipkin_VQE(v[i], w, N, maxiter=300)
        E_VQE_nq[i], pt_nq = run_lipkin_VQE(v[i], w, N, maxiter=1000, force_nq_scheme=True, initial_point=pt_nq)
        E_HF[i] = run_HF(v[i], w, N)
        E_RPA[i] = run_RPA(v[i], w, N)

    fig, ax = plt.subplots()
    ax.plot(v, E_DIAG, c="gray", label="Diagonalization")
    ax.scatter(v, E_VQE, marker="x", c="b", label="VQE $q=2$")
    ax.scatter(v, E_VQE_nq, marker="x", c="r", label=rf"VQE $q=4$")
    ax.plot(v, E_HF, marker="+", markersize=4, label="HF")
    ax.plot(v, E_RPA, label="RPA")
    ax.set(xlabel=r"$V/\epsilon$", ylabel=r"$E_0 / \epsilon$")
    ax.legend()
    plot_utils.save(f"N{N}_lipkin_vary_V")
    plt.show()

def plot_diff_heatmap(N=4, load=False, filename="test", method="VQE", vocal=False):
    n_pts = 50
    v_range = np.linspace(0,1, n_pts)
    w_range = np.linspace(0,1, n_pts)

    run_lipkin_get_E = {
        "HF": run_HF,
        "RPA":run_RPA,
        "VQE": run_lipkin_VQE,
    }

    assert method in list(run_lipkin_get_E.keys()), f"{method = } is not valid. Must be one of {run_lipkin_get_E.keys()}"

    v_grid, w_grid = np.meshgrid(v_range, w_range, indexing="ij")
    E_exact, E_calc = np.zeros_like(v_grid), np.zeros_like(v_grid)

    if not load:
        for i in range(n_pts):
            for j in range(n_pts):
                v, w = v_grid[i, j], w_grid[i,j]
                E_exact[i,j] = run_lipkin_DIAG(v, w, N=N)
                E_calc[i,j] = run_lipkin_get_E[method](v, w, N=N)
                if vocal:
                    print(f"done {(j+1) + i*n_pts}, {v = :.4f}, {w = :.4f}, diff = {np.abs(E_exact[i,j]-E_calc[i,j])}")
            
        with open(f"{filename}.npy", "wb") as file:
            np.save(file, v_grid)
            np.save(file, w_grid)
            np.save(file, E_exact)
            np.save(file, E_calc)

    if load:
        with open(f"{filename}.npy", "rb") as file:
            v_grid = np.load(file)
            w_grid = np.load(file)
            E_exact = np.load(file)
            E_calc = np.load(file)
            
    diff = np.abs((E_exact - E_calc)/E_exact) * 100

    fig, ax = plt.subplots()
    cnt = ax.contourf(v_grid, w_grid, diff, levels=np.linspace(0,100,100, endpoint=True))
    cbar = fig.colorbar(cnt, ax=ax)

    for c in cnt.collections:
        c.set_edgecolor("face")

    cbar.set_label(rf"{method} \% relative error", rotation=90)
    cbar.set_ticks(cbar.get_ticks().astype(int))
    ax.set(xlabel=r"$V/\epsilon$", ylabel=r"$W/\epsilon$")
    plot_utils.save(filename)
    plt.show()

if __name__ == '__main__':
    # plot_v_vs_E(N=2)
    # plot_v_vs_E(N=4)
    plot_diff_heatmap(load=False, filename="50pts_vw_grid_VQE", vocal=True)
    # plot_diff_heatmap(method="HF", filename="50pts_vw_grid_HF")
    # plot_diff_heatmap(method="RPA", filename="50pts_vw_grid_RPA")
