"""
Created on Tue Sep 24 23:42:34 2024

@author: Morufdeen ATILOLA

"""

import numpy as np

def build_Jacobian(v, Y_bus, num_buses):
    """
    Constructs the Jacobian matrix for power flow analysis.

    Parameters:
    v : ndarray
        Voltage at each bus (complex values).
    Y_bus : ndarray
        Admittance matrix (Y_bus).
    num_buses : int
        The number of buses in the system.

    Returns:
    Jacobian : ndarray
        The constructed Jacobian matrix.
    """
    # Initialize the Jacobian sub-matrices with zeros
    J1 = np.zeros((num_buses - 1, num_buses - 1))
    J2 = np.zeros((num_buses - 1, num_buses - 1))
    J3 = np.zeros((num_buses - 1, num_buses - 1))
    J4 = np.zeros((num_buses - 1, num_buses - 1))

    for i in range(1, num_buses):  # Start from bus 2 (index 1 in Python)
        V_i = np.abs(v[i])
        delta_i = np.angle(v[i])
        for j in range(1, num_buses):  # Start from bus 2 (index 1 in Python)
            if i == j:
                for k in range(num_buses):
                    if k != i:
                        V_k = np.abs(v[k])
                        delta_k = np.angle(v[k])
                        Y_ik = np.abs(Y_bus[i, k])
                        theta_ik = np.angle(Y_bus[i, k])

                        J1[i-1, j-1] -= V_i * V_k * Y_ik * np.sin(delta_i - delta_k - theta_ik)
                        J2[i-1, j-1] += V_k * Y_ik * np.cos(delta_i - delta_k - theta_ik)
                        J3[i-1, j-1] += V_i * V_k * Y_ik * np.cos(delta_i - delta_k - theta_ik)
                        J4[i-1, j-1] += V_k * Y_ik * np.sin(delta_i - delta_k - theta_ik)

                J2[i-1, j-1] += 2 * V_i * np.abs(Y_bus[i, i]) * np.cos(np.angle(Y_bus[i, i]))
                J4[i-1, j-1] -= 2 * V_i * np.abs(Y_bus[i, i]) * np.sin(np.angle(Y_bus[i, i]))
            else:
                V_j = np.abs(v[j])
                delta_j = np.angle(v[j])
                Y_ij = np.abs(Y_bus[i, j])
                theta_ij = np.angle(Y_bus[i, j])

                J1[i-1, j-1] = V_i * V_j * Y_ij * np.sin(delta_i - delta_j - theta_ij)
                J2[i-1, j-1] = V_i * Y_ij * np.cos(delta_i - delta_j - theta_ij)
                J3[i-1, j-1] = -V_i * V_j * Y_ij * np.cos(delta_i - delta_j - theta_ij)
                J4[i-1, j-1] = V_i * Y_ij * np.sin(delta_i - delta_j - theta_ij)

    # Concatenate the Jacobian sub-matrices to form the full Jacobian matrix
    Jacobian = np.block([[J1, J2], [J3, J4]])

    return Jacobian
