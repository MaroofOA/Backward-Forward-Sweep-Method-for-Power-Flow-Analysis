"""
Created on Tue Sep 24 23:44:47 2024

@author: Morufdeen ATILOLA

"""
import numpy as np

def calculate_system_loss(num_buses, line_data, v, Z_base):
    """
    Calculate total active and reactive power losses in the system.

    Parameters:
    num_buses : int
        The number of buses in the system.
    line_data : ndarray
        The line data, where each row contains [from_bus, to_bus, R, X].
    v : ndarray
        Voltage at each bus (complex values).
    Z_base : float
        The base impedance value.

    Returns:
    total_active_loss : float
        Total active power loss in the system.
    total_reactive_loss : float
        Total reactive power loss in the system.
    """
    # Initialize total active and reactive power losses
    total_active_loss = 0
    total_reactive_loss = 0

    # Create line impedance matrix to be used for power loss calculation
    Z_line = np.zeros((num_buses, num_buses), dtype=complex)  # Initialize the line impedance matrix with zeros
    for i in range(line_data.shape[0]):  # Iterate over each line in the line_data
        from_bus = int(line_data[i, 0]) - 1  # Convert to zero-based indexing
        to_bus = int(line_data[i, 1]) - 1  # Convert to zero-based indexing
        R = line_data[i, 2]  # Extract the resistance of the line (already in p.u.)
        X = line_data[i, 3]  # Extract the reactance of the line (already in p.u.)
        Z_line[from_bus, to_bus] = (R + 1j * X) / Z_base  # Set the impedance (R + jX) in the impedance matrix
        Z_line[to_bus, from_bus] = (R + 1j * X) / Z_base  # Assuming symmetrical impedance between the buses

    # Initialize the branch current matrix
    I_branch = np.zeros((num_buses, num_buses), dtype=complex)  # To store the branch currents between buses

    # Calculate branch currents
    for i in range(num_buses):
        for j in range(num_buses):
            if Z_line[i, j] != 0:  # If there's an impedance between bus i and bus j
                I_branch[i, j] = (v[i] - v[j]) / Z_line[i, j]  # Calculate current using Ohm's law

    # Calculate system loss
    for i in range(num_buses):
        for j in range(num_buses):
            if Z_line[i, j] != 0 and i < j:  # Only sending to receiving bus loss is considered
                branch_loss = abs(I_branch[i, j]) ** 2 * Z_line[i, j]  # Power loss in the branch
                total_active_loss += np.real(branch_loss)
                total_reactive_loss += np.imag(branch_loss)
                print(f'Loss in branch {i+1}-{j+1}: {np.real(branch_loss):.4f} p.u. + {np.imag(branch_loss):.4f}j p.u.')

    return total_active_loss, total_reactive_loss

