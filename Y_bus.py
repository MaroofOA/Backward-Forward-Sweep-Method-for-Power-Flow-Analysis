"""
Created on Mon Aug  5 10:40:38 2024

@author: Morufdeen ATILOLA

"""
import numpy as np

def generate_Y_bus(bus_data, line_data):
    
    prompt = (
        "Ensure the argument data i.e, bus and line data are in this format:\n"
              "Bus data: column1 - bus index, column2 - real power (P), and column3 - reactive power (Q)\n"
              "Line data: column1 - sending bus index, column2 - receiving bus index, column3 - resistance (R), and reactance (X)"
              )
    
    print(prompt)
    
    user_input = input("Press Enter to continue or type 'quit' to exit: \n");
    
    if user_input.lower() == 'quit':
        print("Exiting the function as per user request probably because the data does not conform with the requirement")
        
        return 
    
    # Initialize the Y_bus matrix
    Y_bus = np.zeros((len(bus_data), len(bus_data)), dtype = complex)

    # Add self admittance terms
    for i in range(len(bus_data)):
        # Select rows where sending and receiving data matches the current bus
        connected_buses = np.where((line_data[:, 0] == i+1) | (line_data[:, 1] == i+1))[0]
        
        # Extract the resistance and reactance values
        resistance = line_data[connected_buses, 2]
        reactance = line_data[connected_buses, 3]
        
        # Calculate the impedance and admittance
        impedance = resistance + 1j * reactance
        admittance = 1 / impedance
        
        # Sum up all the individual admittance
        Y_bus[i, i] = np.sum(admittance)

    # Add mutual admittance terms
    for i in range(len(line_data)):
        sending_bus = line_data[i, 0] - 1
        receiving_bus = line_data[i, 1] - 1
        resistance = line_data[i, 2]
        reactance = line_data[i, 3] * 1j
        
        # Calculate the impedance and admittance
        impedance = resistance + reactance
        admittance = 1 / impedance
        
        # Update the Y_bus matrix
        Y_bus[int(sending_bus), int(receiving_bus)] = -1 * admittance
        Y_bus[int(receiving_bus), int(sending_bus)] = -1 * admittance

    # Display and download the Y_bus
    print(Y_bus)
    print(np.diag(Y_bus))

    # Create a dynamic filename for the file downloads
    filename_csv = f'Y_bus_{len(bus_data)}bus.csv'
    np.savetxt(filename_csv, Y_bus, delimiter=',',  fmt='%.5f')

    filename_txt = f'Y_bus_{len(bus_data)}bus.txt'
    np.savetxt(filename_txt, Y_bus, delimiter=' ', fmt='%.5f')
