function Jacobian = build_Jacobian(v, Y_bus, num_buses)
    J1 = zeros(num_buses - 1);
    J2 = zeros(num_buses - 1);
    J3 = zeros(num_buses - 1);
    J4 = zeros(num_buses - 1);

    for i = 2:num_buses
        V_i = abs(v(i));
        delta_i = angle(v(i));
        for j = 2:num_buses
            if i == j
                for k = 1:num_buses
                    if k ~= i
                        V_k = abs(v(k));
                        delta_k = angle(v(k));
                        Y_ik = abs(Y_bus(i, k));
                        theta_ik = angle(Y_bus(i, k));

                        J1(i-1, j-1) = J1(i-1, j-1) - V_i * V_k * Y_ik * sin(delta_i - delta_k - theta_ik);
                        J2(i-1, j-1) = J2(i-1, j-1) + V_k * Y_ik * cos(delta_i - delta_k - theta_ik);
                        J3(i-1, j-1) = J3(i-1, j-1) + V_i * V_k * Y_ik * cos(delta_i - delta_k - theta_ik);
                        J4(i-1, j-1) = J4(i-1, j-1) + V_k * Y_ik * sin(delta_i - delta_k - theta_ik);
                    end
                end
                J2(i-1, j-1) = J2(i-1, j-1) + 2 * V_i * abs(Y_bus(i, i)) * cos(angle(Y_bus(i, i)));
                J4(i-1, j-1) = J4(i-1, j-1) - 2 * V_i * abs(Y_bus(i, i)) * sin(angle(Y_bus(i, i)));
            else
                V_j = abs(v(j));
                delta_j = angle(v(j));
                Y_ij = abs(Y_bus(i, j));
                theta_ij = angle(Y_bus(i, j));

                J1(i-1, j-1) = V_i * V_j * Y_ij * sin(delta_i - delta_j - theta_ij);
                J2(i-1, j-1) = V_i * Y_ij * cos(delta_i - delta_j - theta_ij);
                J3(i-1, j-1) = -V_i * V_j * Y_ij * cos(delta_i - delta_j - theta_ij);
                J4(i-1, j-1) = V_i * Y_ij * sin(delta_i - delta_j - theta_ij);
            end
        end
    end
    
    Jacobian = [J1, J2; J3, J4];
end
