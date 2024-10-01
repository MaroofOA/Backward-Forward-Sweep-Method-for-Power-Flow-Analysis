function Y_bus = build_Y_bus(line_data, num_buses, Z_base)
        % Helper function to construct Y_bus matrix
        % Construct the admittance matrix (Y_bus) based on line_data
        % Initialize the Y_bus matrix
        Y_bus = zeros(num_buses, num_buses); % Use a zero matrix for initialization
        % Calculate the admittance matrix
        for i = 1:num_buses
            connected_buses = (line_data(:, 1) == i | line_data(:, 2) == i);
            R = line_data(connected_buses, 3); % Resistance
            X = line_data(connected_buses, 4); % Reactance

            Z = (R + 1j * X) / Z_base; % Impedance
            Y = 1 ./ Z; % Admittance

            % Add self admittance terms
            Y_bus(i, i) = sum(Y);
        end

        % Add mutual admittance terms
        for i = 1:size(line_data, 1)
            from_bus = line_data(i, 1);
            to_bus = line_data(i, 2);
            R = line_data(i, 3);
            X = line_data(i, 4) * 1j;
            Z = (R + X) / Z_base;
            Y = 1 / Z;

            % Update the Y_bus matrix for mutual admittance terms
            Y_bus(from_bus, to_bus) = -Y;
            Y_bus(to_bus, from_bus) = -Y;
        end
        
end        