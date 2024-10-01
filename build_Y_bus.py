"""
Created on Tue Sep 24 23:41:05 2024

@author: Morufdeen ATILOLA

"""
import numpy as np

def build_Y_bus(line_data, num_buses, Z_base):
    """
    Constructs the Y_bus (admittance) matrix based on line_data.

    Parameters:
    line_data : array-like
        A 2D array where each row represents a line between two buses.
        Columns represent [from_bus, to_bus, resistance (R), reactance (X)].
    num_buses : int
        The total number of buses in the system.
    Z_base : float
        The base impedance used to normalize the impedances.

    Returns:
    Y_bus : ndarray
        The constructed Y_bus matrix (admittance matrix).
    """
    # Initialize the Y_bus matrix with zeros
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

    # Calculate self-admittance terms
    for i in range(num_buses):
        connected_buses = np.logical_or(line_data[:, 0] == i + 1, line_data[:, 1] == i + 1)
        R = line_data[connected_buses, 2]  # Resistance
        X = line_data[connected_buses, 3]  # Reactance

        Z = (R + 1j * X) / Z_base  # Impedance
        Y = 1 / Z  # Admittance

        # Add self admittance terms
        Y_bus[i, i] = np.sum(Y)

    # Add mutual admittance terms
    for i in range(line_data.shape[0]):
        from_bus = int(line_data[i, 0]) - 1  # MATLAB indices start at 1, Python at 0
        to_bus = int(line_data[i, 1]) - 1
        R = line_data[i, 2]
        X = line_data[i, 3] * 1j
        Z = (R + X) / Z_base
        Y = 1 / Z

        # Update the Y_bus matrix for mutual admittance terms
        Y_bus[from_bus, to_bus] = -Y
        Y_bus[to_bus, from_bus] = -Y

    return Y_bus
