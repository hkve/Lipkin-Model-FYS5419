from qiskit.opflow import PauliSumOp
import numpy as np

def get_hamiltonian_matrix(v, w, N):
    H_exact = None
    if N == 2:
        H_exact = np.zeros((3,3))
        H_exact[0,0] = -1
        H_exact[1,1] = w
        H_exact[2,2] = 1
        H_exact[2,0] = H_exact[0,2] = -v
    elif N == 4:
        H_exact = np.zeros((5,5))
        H_exact[0,0] = -2
        H_exact[1,1] = -1 + 3*w
        H_exact[2,2] = 4*w
        H_exact[3,3] = 1 + 3*w
        H_exact[4,4] = 2

        H_exact[2,0] = H_exact[0,2] = H_exact[2,4] = H_exact[4,2] = np.sqrt(6)*v
        H_exact[3,1] = H_exact[1,3] = -3*v

    return H_exact


def get_hamiltonian(v, w, N):
    sp = 0.5
    t1 = 0.5*(w+v)
    t2 = 0.5*(w-v)
    if N == 2:
        hamiltonian = PauliSumOp.from_list([
          ("ZI", sp),
          ("IZ", sp),
          ("XX", t1),
          ("YY", t2)
        ])
    elif N == 4:
        if w == 0:
            v_prime = v * np.sqrt(6)/2
            hamiltonian = PauliSumOp.from_list([
                ("ZI", -1),
                ("IZ", -1),
                ("XI", -v_prime),
                ("IX", -v_prime),
                ("ZX", -v_prime),
                ("XZ", v_prime)
            ])
        else:
            hamiltonian = PauliSumOp.from_list([("ZIII", sp),
                ("IZII", sp),
                ("IIZI", sp),
                ("IIIZ", sp),
                ("XXII", t1),
                ("XIXI", t1),
                ("XXIX", t1),
                ("IXXI", t1),
                ("IXIX", t1),
                ("IIXX", t1),
                ("YYII", t2),
                ("YIYI", t2),
                ("YYIY", t2),
                ("IYYI", t2),
                ("IYIY", t2),
                ("IIYY", t2),
            ])
    return hamiltonian