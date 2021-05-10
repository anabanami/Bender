import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt


matrix_1 = np.load('matrix_-1.0.npy')

eigenvalues_1, eigenvectors_1 = linalg.eig(matrix_1)

non_zero_smol_imag = [i for i in eigenvalues_1 if abs(np.imag(i)) <= 1e-14 and np.real(i) !=0]
sorted_eigenvalues = sorted(non_zero_smol_imag, key=lambda x: x.real)

# print(f"\nEigenvalues ϵ: -1.0\n{eigenvalues_1}")
# print(f"\n{len(eigenvalues_1) = }\n")

# print(f"\n{len(non_zero_smol_imag) = }")
# print(f"\n{non_zero_smol_imag = }")

print(f"\n{sorted_eigenvalues = }\n")



