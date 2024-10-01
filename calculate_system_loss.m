function [total_active_loss, total_reactive_loss] = calculate_system_loss(num_buses, line_data, v, Z_base)
    % Initialize total active and reactive power losses
    total_active_loss = 0;
    total_reactive_loss = 0;
    
    % Create line impedance matrix to be used for power loss calculation
    Z_line = zeros(num_buses, num_buses) + 1j * zeros(num_buses, num_buses); % Initialize the line impedance matrix with zeros.
    for i = 1:size(line_data, 1) % Iterate over each line in the line_data.
        from_bus = line_data(i, 1); % Get the starting bus index of the line.
        to_bus = line_data(i, 2); % Get the ending bus index of the line.
        R = line_data(i, 3); % Extract the resistance of the line. Already in p.u.
        X = line_data(i, 4); % Extract the reactance of the line. Already in p.u.
        Z_line(from_bus, to_bus) = (R + 1j * X)/Z_base; % Set the impedance (R + jX) in the impedance matrix.
        Z_line(to_bus, from_bus) = (R + 1j * X)/Z_base; % Assuming the line impedance is symmetrical and set opposite entry.
    end
    
    % Initialize the branch current matrix
    I_branch = zeros(num_buses, num_buses); % This will store the branch currents between buses.

    % Calculate branch currents
    for i = 1:num_buses
        for j = 1:num_buses
            if Z_line(i, j) ~= 0 % If there's an impedance between bus i and bus j
                I_branch(i, j) = (v(i) - v(j)) / Z_line(i, j); % Calculate the current from bus i to bus j using Ohm's law.
            end
        end
    end
    
    % Calculate system loss
    for i = 1:num_buses
        for j = 1:num_buses
            if Z_line(i, j) ~= 0 && i < j % Ensure only sending to receiving bus loss is considered
                branch_loss = abs(I_branch(i, j))^2 * Z_line(i, j);
                total_active_loss  = total_active_loss + real(branch_loss);
                total_reactive_loss  = total_reactive_loss + imag(branch_loss);
                fprintf('Loss in branch %d-%d: %.4f p.u. + %.4fj p.u.\n', i, j, real(branch_loss), imag(branch_loss));
            end
        end
    end
end
