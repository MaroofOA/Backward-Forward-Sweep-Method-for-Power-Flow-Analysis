# Backward Forward Sweep Method for Power Flow Analysis

This repository contains implementations of the **Backward Forward Sweep** method for performing power flow analysis on distribution systems in both MATLAB and Python. The code is versatile and can be used for any system model, provided that the input data adheres to the specified format.

## Overview

The main function for this method computes the voltage profile of buses in a radial distribution system. It requires two primary inputs: load data and line data. The load data should specify the bus index along with the real and reactive power for each bus, while the line data should include information on the resistance and reactance of the lines connecting the buses.

### Functionality

The function includes the following features:
- **Data Input Requirements**: 
  - The load data must be structured such that it has three columns: bus index, real power (P), and reactive power (Q).
  - The line data must have four columns: sending bus index, receiving bus index, resistance (R), and reactance (X).
  
- **Customizable Parameters**: Users can specify a constant slack bus voltage, a convergence tolerance for iteration, and a maximum number of iterations allowed for the calculations. Default values for these parameters are provided, which can be overridden when calling the function.

### Guidance for Use

1. **Input Format**: Ensure that the input data (i.e., load and line data) are formatted correctly:
   - **Load Data**: 
     - Column 1: Bus index
     - Column 2: Real power (P)
     - Column 3: Reactive power (Q)
   - **Line Data**: 
     - Column 1: Sending bus index
     - Column 2: Receiving bus index
     - Column 3: Line resistance (R)
     - Column 4: Line reactance (X)

2. **Function Call**: To execute the function, you can provide the load and line data as arguments, along with optional parameters for slack bus voltage, tolerance, and maximum iterations. If you skip any of the optional parameters, the function will use the predefined default values.

3. **Example Usage**: The prompt provides an example of how to call the function. For instance:
   - **Python**:
     ```python
     v, iteration = bfs_method(load_data, line_data, slack_bus_voltage=1.5, tolerance=1e-4)
     ```
   - **MATLAB**:
     ```matlab
     [v, iteration] = bfs_method(load_data, line_data, 1.5, 1e-4);
     ```

4. **Exiting the Function**: Users are prompted to press Enter to continue or type 'quit' to exit the function. This allows for flexibility in how users interact with the function.

### Expected Outputs

Upon successful execution, the function returns the following outputs:
- Voltage after each iteration
- Number of iterations
- Time taken for each iteration
- Average time taken for all runs
- Total active and reactive power loss
- Substation active and reactive power
- Graph of voltage magnitude and angle
- Single line diagram of the system model
- Graph of the relationship between maximum error and computational time
- .csv and .txt file of the voltage profile

This provides a comprehensive tool for conducting power flow analysis in distribution networks, supporting both MATLAB and Python users.

## Files

- `bfs_method.py`: Python code for the Backward Forward Sweep power flow analysis.
- `bfs_method.m`: MATLAB code for the Backward Forward Sweep power flow analysis.
