% Author: Kenji Kashima
% Date  : 2025/09/10
% Note  : Optimization Toolbox is required

clear;close all; rng(3); % random seed

% parameter setting
k_bar = 50;
k_bar_i = 1000;
sigma_param = 1.0; 

% transition probability matrix (P0)
P = [1/3,    1/3,     0,     0; 
      0  ,    1/3,   1/3,     0;
      0  ,    1/3,   1/3,   1/3;
      2/3,      0,   1/3,   2/3];

p_stationary = stationary_distribution(P);
disp('p_stationary for P^0:');
disp(p_stationary);

% accumulated transition probability
[m,n] =size(P);
P_accum = cumsum(P,1);

p_list = zeros(1,k_bar); 
p_list(1)=4; % start at 4
for i=2:k_bar
    u = rand;
    T = P_accum(:,p_list(i-1)); %transition probability
    for j = 1:m
        if u <= T(j)
             p_list(i) = j;
             break
        end
    end
end

figure('Name','Figure10.2(b)'); hold on; grid on;
stairs(p_list);
yticks(1:4);
xlim([0,k_bar]);
ylim([0.8,4.2]);
xlabel('$k$','Fontsize',16,'Interpreter','latex')
title('$P^0$','Fontsize',16,'Interpreter','latex')

%% inverse reinforcement learning

beta = 0.8;
invbeta = 1/beta;
cost_val = [1, 2, 3, 4] * sigma_param; 

% solve 10.23 L_KL
options = optimoptions('fminunc', 'Algorithm','quasi-newton',...
    'OptimalityTolerance',1e-10, 'FunctionTolerance',1e-14,...
    'MaxIterations',1000, 'MaxFunctionEvaluations',5000);
v_init_kl = ones(4,1)*5; % Initialization for minimization
V_star_optimal = fminunc(@(v_star)L_KL(v_star, beta, cost_val, P), v_init_kl, options); % minimization of L_KL

disp("Value function：");
disp(V_star_optimal);
z_opt = exp(-beta * V_star_optimal); % Calculate Z* from V*

% compute P_opt from Z*
P_opt = zeros(size(P));
for r_idx = 1:m 
    for c_idx = 1:n 
        P_opt(r_idx,c_idx) = P(r_idx,c_idx) * z_opt(r_idx) / (P(:,c_idx)' * z_opt); 
    end
end

fprintf("sigma: %.1f\nPpi:\n", sigma_param);
disp(P_opt);

p_opt_stationary = stationary_distribution(P);
disp("p_stationary for P_opt：");
disp(p_opt_stationary);


% accumulate P
P_accum = cumsum(P_opt, 1);

% simulation
p_list = zeros(k_bar_i+1, 1, 'int8');
p_list(1) = 4; % initial state
a_counts = [0; 0; 0; 1]; 
b_counts = [0; 0; 0; 0]; 
inv_l_hist = zeros(4, k_bar);

% constrains (you can change lb, to move the baseline)
options_v_fmincon = optimoptions('fmincon','Display','off'); 
lb_v = [12; 12; 12; 12]; % lower bound constrains

for k_i = 1:k_bar_i 
    % maximize the likelyhood L_IRL 
    v_irl_init = [1; 2; 3; 4]; % Initialization for minimization
    V_star_estimated_irl = fmincon(@(v_star_irl)L_IRL(v_star_irl, beta, a_counts, b_counts, P), v_irl_init,... % mimization of L_IRL
        [],[],[],[],lb_v,[],[],options_v_fmincon);
    
    inv_v_opt_irl = V_star_estimated_irl; 
    inv_z_opt_irl = exp(-beta * inv_v_opt_irl); 
    inv_l_estimated = -log(inv_z_opt_irl.^invbeta ./ (P' * inv_z_opt_irl)); 
    inv_l_hist(:,k_i) = inv_l_estimated;
    
    % state transition
    u_rand = rand(); 
    T_accum = P_accum(:, p_list(k_i));  
    for j_state = 1:m 
        if u_rand <= T_accum(j_state)
            p_list(k_i+1) = j_state;
            break;
        end
    end
    
    % update count
    a_counts(p_list(k_i+1)) = a_counts(p_list(k_i+1)) + 1;
    b_counts(p_list(k_i)) = b_counts(p_list(k_i)) + 1;
end

% plot Figure 10.4a
figure('Name','Figure10.4(a)');
stairs(0:k_bar_i, double(p_list) );
ylim([0.8, 4.2]);
xlim([0, 50]);
xticks([0, 50]);
xlabel('k');
yticks(1:4);
grid on;

% plot Figure 10.4b
figure('Name','Figure10.4(b)');
hold on;
for i_plot = 1:4 
    plot(inv_l_hist(i_plot,:)-inv_l_hist(1,:)+1, 'DisplayName', sprintf('$\\ell_{%d}$', i_plot));
end
ylim([0, 6]);
xlim([0, k_bar_i]);
xlabel('k');
legend('Interpreter','latex');
grid on;


% functions

function p_stationary = stationary_distribution(P)
    [V, D] = eig(P);
    [~, index] = max(diag(D));
    eigenvector_for_1 = V(:, index);
    sum_of_elements = sum(eigenvector_for_1);
    p_stationary = eigenvector_for_1 / sum_of_elements;
end

% Loss function for KL control, and it now optimizes V* directly
function sum_sq_residual = L_KL(V_star_var, beta_param, cost_val_param, P0_matrix) 
    term_log = log(P0_matrix' * exp(-beta_param * V_star_var));
    residuals = V_star_var - cost_val_param(:) + term_log;
    sum_sq_residual = sum(residuals.^2);
end

% Loss Function for IRL
function neg_log_likelihood = L_IRL(V_star_var, beta_param, a_counts_param, b_counts_param, P0_matrix) 
    term_log = log(P0_matrix' * exp(-beta_param * V_star_var));
    neg_log_likelihood = beta_param * a_counts_param' * V_star_var + b_counts_param' * term_log;
end