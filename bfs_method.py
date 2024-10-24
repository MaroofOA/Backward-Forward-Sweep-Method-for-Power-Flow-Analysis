"""
Created on Mon Aug  5 10:40:10 2024

@author: Morufdeen ATILOLA

"""

# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
# import networkx as nx # for drawing the system network with flow arrow

# Define the function for the backward_forward method for distribution network analysis. 
def bfs_method(load_data, line_data, slack_bus_voltage=1, tolerance=1e-6, max_iter = 100, runs = 10):
    # arguments includes the constant slack bus voltage, convergence value, and max. iterations, all of which can be changed when the function is called
    prompt = (
        "Ensure the argument data i.e., load and line data are in this format:\n"
        "load data: column1 - bus index, column2 - real power (P), and column3 - reactive power (Q)\n"
        "line data: column1 - sending bus index, column2 - receiving bus index, column3 - resistance (R), and reactance (X)\n"
        "\n"
        "The default values are slack_bus_voltage = 1.5, tolerance = 1e-6, and max_iter = 100\n"
        "If you decide to change any of the constant argument values, such as slack_bus_voltage, tolerance, or max_iter, just pass the new value in their respective position\n"
        "If you will jump any of the order, kindly put the constant value of the one you are jumping in its position and input the new value of the argument you are changing\n"
        "For example: [v, iteration] = bfs_method(load_data, line_data, 1.5, 1e-4), this is because, I want to keep the slack bus the same but want to change the tolerance\n"
        "I did not put the value of max_iter because, it is the last one and I want to keep it the same.\n"
        )
    
    print(prompt)
    
    user_input = input("Press Enter to continue or type 'quit' to exit: \n");
    
    if user_input.lower() == 'quit':
        print("Exiting the function as per user request probably because the data does not conform with the requirement")
        iteration = []  # Return an empty array
        v = []
        return v, iteration
    
    computation_times = []
    
    for run in range(runs):
        # System base values
        V_base = 12.66  # Nominal voltage in kV
        S_base = 1  # Base power in MVA
        Z_base = (V_base ** 2) / S_base  # Base impedance in ohms
        
        # Determine the number of buses from the load data
        num_buses = load_data.shape[0]
        
        # Initialize bus voltages
        v = np.ones(num_buses, dtype=complex) # let all the initial voltage for all the buses be 1.0 p.u.
        v[0] = slack_bus_voltage  # Set the slack bus voltage
        
        # Create load vectors
        P_load = load_data[:, 1] / 1000  # Convert kW MW and to per-unit
        Q_load = load_data[:, 2] / 1000  # Convert kVar to MVar and to per-unit
        S_load = P_load + 1j * Q_load  # Complex power (S = P + jQ)
            
        # Create line impedance matrix
        Z_line = np.zeros((num_buses, num_buses), dtype=complex) #  Initializes the line impedance matrix with zeros.
        for i in range(line_data.shape[0]): # Iterates over each line in the line_data.
            from_bus = int(line_data[i, 0] - 1) # Gets the starting bus index of the line. (the -1 is for the numbering of the bus to conform with python zeroth starting rule)
            to_bus = int(line_data[i, 1] - 1) # Gets the ending bus index of the line.
            R = line_data[i, 2] # Extracts the resistance of the line. Already in p.u.
            X = line_data[i, 3] # Extracts the reactance of the line. Already in p.u.
            Z_line[from_bus, to_bus] = (R + 1j * X)/Z_base # Sets the impedance (R + jX) in the impedance matrix and convert to p.u.
            Z_line[to_bus, from_bus] = (R + 1j * X)/Z_base  # Assuming the line impedance is symmetrical and sets oppoite entry.
        
        # Initialize empty list to store maximum errors and iteration times
        max_voltage_errors = []
        cumulative_iter_times = []
        
        # Measure the computational time for BFS method only
        start_time = time.time()
            
        # Backward-Forward Sweep method
        for iteration in range(max_iter): # Iterates up to max_iter times to perform the BFS algorithm.
            
            # Start timing for the iteration
            iter_start_time = time.time()
        
            V_old = v.copy() #  Copies the current bus voltages to V_old for convergence checking.
            
            # Backward Sweep
            I_load = np.zeros(num_buses, dtype=complex) # Initializes the load currents with zeros.
            I_branch = np.zeros((num_buses, num_buses), dtype=complex) # Initializes the branch currents with zeros.
            
            # Calculate load currents
            for i in range(num_buses): # Iterates over each of the buses
                I_load[i] = np.conj(S_load[i] / v[i])  # Calculate load current at bus i
            # print(f"Iteration {iteration + 1} - Load Currents: {I_load}") # print the load current at that iteration
                
            # Calculate branch currents explicitly
            for i in range(num_buses - 1, 0, -1):  # Iterate backward from the last bus to the first bus
                for j in range(num_buses): # Iterates over all buses to sum the downstream currents.
                    if Z_line[j, i] != 0:  # Check if there's an impedance (i.e., connection)
                        I_branch[j, i] = I_load[i] + I_branch[i].sum()  # I_ij = I_j + sum of all currents leaving j, Adds the downstream current to the current bus load current.
            # print(f"Iteration {iteration + 1} - Branch Currents: {I_branch}") print the branch current at that iteration
                                    
            # Forward Sweep
            v[0] = V_old[0]  # Voltage at the slack bus remains the same
    
            # Iterate over each bus to update voltages
            for i in range(1, num_buses):  # Start from bus 1 (excluding slack bus)
                for j in range(i):  # Iterate over all predecessor buses to find the connection
                    if Z_line[j, i] != 0:  # There is an impedance between buses j and i
                        # Calculate the voltage for bus i based on bus j
                        current = I_branch[j, i]  # Current in the branch from j to i
                        impedance = Z_line[j, i]  # Impedance of the branch from j to i
                        v[i] = v[j] - impedance * current  # Update the voltage of bus i
    
                        """ 
                        # Print debug information
                        print(f"Updating bus {i+1} using bus {j+1}:")
                        print(f"Current I_{j+1}{i+1}: {current}")
                        print(f"Impedance Z_{j+1}{i+1}: {impedance}")
                        print(f"Updated voltage V_{i+1}: {v[i]}")
                        
                        """
                        # Break after the first update to ensure each bus is updated only once
                        break
            
                    
            # Print voltages after the forward sweep
            print(f"Iteration {iteration + 1} - Voltages: {v}")
            
            # End timing the iteration
            iter_time = time.time()
            
            # Check for convergence
            max_diff = np.max(np.abs(v - V_old))
            print(f"Iteration {iteration + 1}: max voltage difference = {max_diff}")
            
            # Store max voltage difference and cumulative iteration time
            max_voltage_errors.append(max_diff)
            
            if iteration == 0:
                cumulative_iter_times.append(iter_time)
            else:
                cumulative_iter_times.append(cumulative_iter_times[-1] + iter_time)
            
            # Print max voltage difference at an iteration and its respective computation time till that iteration.
            print(f"Iteration {iteration+1}: max voltage difference = {max_diff:.10f}")
            print(f"Time taken till {iteration+1} iteration: {iter_time:.4f} seconds")
            
            if max_diff <= tolerance: # Checks if the maximum voltage difference between iterations is less than the tolerance.
                break # Breaks the loop if convergence is achieved.
    
        # End computational time measurement
        end_time = time.time()
        computation_time = end_time - start_time
        computation_times.append(computation_time)

    # To handle only the iterations performed
    actual_iterations = iteration + 1  # Number of iterations performed until convergence
    max_voltage_errors = max_voltage_errors[:actual_iterations]
    cumulative_iter_times = cumulative_iter_times[:actual_iterations]

    # Prepare results and print them
    results = [] # Initialize an empty list to store the results.
    print(f"Converged in {iteration+1} iterations.") # Print the number of iterations it took to converge.
    
    print("Bus Voltages:") # Print a header for the bus voltages.

    for i, voltage in enumerate(v): # Iterate over each bus voltage.
        magnitude = abs(voltage) # Calculate the magnitude of the voltage.
        angle = np.angle(voltage, deg=True)  # Calculate the angle of the voltage in degrees.
        rectangular_form = f"{voltage.real:.4f} + {voltage.imag:.4f}j"
        polar_form = f"{magnitude:.4f} ∠ {angle:.2f}°"
        print(f"Bus {i+1}: Rectangular form: {rectangular_form} p.u., Polar form: {polar_form}") # Print the voltage in both forms.
        results.append((i+1, voltage, polar_form)) # Append the bus number, original voltage, and polar form to the results list.
        
    # Calculate system loss
    total_active_loss = 0
    total_reactive_loss = 0
    
    for i in range(num_buses):
        for j in range(num_buses):
            if Z_line[i, j] != 0 and i < j:
                branch_loss = np.abs(I_branch[i, j])**2 * Z_line[i, j]
                total_active_loss += branch_loss.real
                total_reactive_loss += branch_loss.imag
                print(f"Loss in branch {i+1}-{j+1}: {branch_loss.real:.4f} p.u. + {branch_loss.imag:.4f}j p.u.")
    
    print(f"Converged in {iteration+1} iterations.")
    
    print(f"Total active power loss: {total_active_loss:.4f} p.u. = {1000*total_active_loss: .4f} kW")
    print(f"Total reactive power loss: {total_reactive_loss:.4f} p.u.  = {1000*total_reactive_loss: .4f} kVAR")
    
    # Calculate substation power
    substation_active_power = np.sum(P_load) + total_active_loss
    substation_reactive_power = np.sum(Q_load) + total_reactive_loss
    
    print(f"Substation active power: {substation_active_power:.4f} p.u. = {substation_active_power:.4f} MW")
    print(f"Substation reactive power: {substation_reactive_power:.4f} p.u. = {substation_reactive_power:.4f} MVAR")
    
    # Find minimum and maximum voltages and their corresponding bus indices
    min_voltage = np.min(np.abs(v))
    min_index = np.argmin(np.abs(v))  # Index of the bus with minimum voltage
    
    max_voltage = np.max(np.abs(v))
    max_index = np.argmax(np.abs(v))  # Index of the bus with maximum voltage
    
    # Print the minimum and maximum voltages along with the bus indices
    print(f'Minimum voltage = {min_voltage:.4f} pu at bus {min_index + 1}')  # Adding 1 for 1-based indexing
    print(f'Maximum voltage = {max_voltage:.4f} pu at bus {max_index + 1}')  # Adding 1 for 1-based indexing
    
    # Create a formatted string representation of the computation times so as to return it as a formated list of strings in 4dp.
    formatted_times = [f"{time:.4f}" for time in computation_times]
    
    # Print the formatted list of computation times
    print(f"Computation times for each run: {formatted_times}")
    
    # Calculate and print the average computation time
    average_time = sum(computation_times) / len(computation_times)
    print(f"Average computation time for the model after {runs} runs: {average_time:.4f} seconds")
    
    """
    # Create the graph for the system network
    G = nx.DiGraph()  # DiGraph is used for directed graphs
    G.add_edges_from(line_data[:, 0:2])
    
    # Draw the system network.
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='r', edge_color='b', arrowsize=10, node_size=20)
    
    # Display the graph
    plt.title("System Network Graph")
    plt.show()
    """
    
    # Plot the voltage magnitudes and angle
    voltage_magnitudes = np.abs(v)
    voltage_angle = np.angle(v)
    bus_indices = np.arange(1, num_buses + 1)

    # Create a figure with 2 subplots
    plt.figure(figsize=(12, 8))
    
    # First subplot for Voltage Magnitude
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(bus_indices, voltage_magnitudes, marker='o', linestyle='-', color='b', label='Voltage Magnitude')
    plt.xlabel('Bus Index')
    plt.ylabel('Voltage Magnitude (p.u.)')
    plt.title('Bus Voltage Magnitudes')
    plt.grid(True)
    plt.legend()
    
    # Second subplot for Voltage Angle
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(bus_indices, voltage_angle, marker='o', linestyle='-', color='r', label='Voltage Angle')
    plt.xlabel('Bus Index')
    plt.ylabel('Voltage Angle (radian)')
    plt.title('Bus Voltage Angles')
    plt.grid(True)
    plt.legend()
    
    # Show the combined plots
    plt.tight_layout()
    plt.show()
    
    # Plot the maximum voltage error vs. computation time till iteration
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_iter_times, max_voltage_errors, marker='o', linestyle='-', color='b')
    plt.xlabel('Computation Time (seconds)')
    plt.ylabel('Maximum Error')
    plt.title('Maximum Error vs. Computation Time')
    plt.grid(True)  # Add grid if desired
    plt.show()
    
    # Write results to a text file with UTF-8 encoding
    with open(f'bus_voltage_{len(load_data)}_bus.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write(f"Converged in {iteration} iterations.\n")
        txt_file.write("Bus Voltages:\n")
        txt_file.write("Bus Number\tRectangular Form\tPolar Form\n")
        for result in results:
            txt_file.write(f"{result[0]}\t{result[1].real:.4f} + {result[1].imag:.4f}j\t{result[2]}\n")
    
    # Write results to a CSV file with UTF-8 encoding
    with open(f'bus_voltage_{len(load_data)}_bus.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Bus Number", "Rectangular Form", "Polar Form"])
        for result in results:
            csv_writer.writerow([result[0], f"{result[1].real:.4f} + {result[1].imag:.4f}j", result[2]])

    return v, iteration+1