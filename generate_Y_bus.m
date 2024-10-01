% Create a function named Y_bus
function Y_bus = generate_Y_bus(bus_data, line_data)

    prompt = {'Ensure the argument data i.e, bus and line data are in this format:'
                   'bus data: column1 - bus index, column2 - real power (P), and column3 - reactive power (Q)'
                   'line data: column1 - sending bus index, column2 - receiving bus index, column3 - resistance (R), and reactance (X)'};
    disp(prompt);
    user_input = input('Press Enter to continue or type ''quit'' to exit; ', 's');

     % Check if the user wants to quit
    if strcmp(user_input, 'quit')% strcmp is the MATLAB function for comparing two string  first to last
        disp('Exiting the function as per user request probably because the data does not conform with the requirement');
        Y_bus = []; % Return an empty array
        return;
    end 
    
    % Initialize the Y_bus matrix
    num_buses = length(bus_data); % Number of buses
    Y_bus = zeros(num_buses, num_buses); % Use a zero matrix for initialization

    % Calculate the admittance matrix
    for i = 1:num_buses
        % Select rows where sending and receiving data matches the current bus
        connected_buses = (line_data(:, 1) == i | line_data(:, 2) == i);
        
        % Extract the resistance and reactance values
        resistance = line_data(connected_buses, 3); % Get the resistance (R)
        reactance = line_data(connected_buses, 4); % Get the reactance (X)
        
        % Calculate the impedance and admittance
        impedance = resistance + 1j * reactance; % (R + jX)
        admittance = 1 ./ impedance; % Get the admittance
        
        % Add self admittance terms
        Y_bus(i, i) = sum(admittance); % Sum up all the individual admittance
    end

    % Add mutual admittance terms
    for i = 1:size(line_data, 1) % Iterate through each line
        sending_bus = line_data(i, 1); % Identify the sending bus
        receiving_bus = line_data(i, 2); % Identify the receiving bus
        resistance = line_data(i, 3); % Get the resistance (R)
        reactance = line_data(i, 4) * 1j; % Get the reactance (X) and multiply by 1j
        impedance = resistance + reactance; % Calculate the impedance
        admittance = 1 / impedance; % Get the admittance
        
        % Update the Y_bus matrix for mutual admittance terms
        Y_bus(sending_bus, receiving_bus) = -admittance; % Yij
        Y_bus(receiving_bus, sending_bus) = -admittance; % Yji
    end
    
 
    % Display and save the Y_bus matrix
    disp('Y_bus Matrix:');
    disp(Y_bus);
    disp('Self Admittance Values:');
    disp(diag(Y_bus)); % Display the self admittance values alone

    % Create dynamic filenames for the file downloads
    filename_csv = sprintf('Y_bus_%d_bus.csv', num_buses);
    csvwrite(filename_csv, Y_bus); % Save the CSV copy of the Y_bus
    filename_txt = sprintf('Y_bus_%d_bus.txt', num_buses);
    dlmwrite(filename_txt, Y_bus, ' '); % Save the TXT file copy of the Y_bus
end
