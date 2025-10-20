% Author: Kenji Kashima
% Date  : 2025/09/01

clear;close all; rng(1); % random seed

set(0, 'DefaultTextInterpreter', 'latex');       
set(0, 'DefaultLegendInterpreter', 'latex');     
set(0, 'DefaultAxesTickLabelInterpreter', 'latex'); 

% Parameters setup
k_bar = 60;  % total steps

% System matrices
A = [-0.39, -0.67, -0.34;
      0.71, -0.51,  0.11;
     -0.46, -0.35, -0.12];  % System matrix A
B_v = [0; 1; 0];           % Input matrix B_v
C = [1, 0, 0];             % Output matrix C

% Initial state mean
mu_0 = [1; 1; 1];          % Mean of the initial state
sigma_0 = eye(3);          % Covariance of the initial state (Identity matrix)

% Noise parameters
n = 3;  % State dimension
v_bounds = [-1, 1];  % Range for process noise v_k (truncated Gaussian)
laplace_scale = 1;   % Scale parameter for Laplace noise w_k

% Generate the data sequence for x and y
x = mvnrnd(mu_0, sigma_0)';  % Initial state x_0 ~ N(mu_0, I)
x_data = zeros(n, k_bar+1);  % Store x_{0:k_bar}
y_data = zeros(1, k_bar+1);  % Store y_{0:k_bar}

x_data(:, 1) = x;

% Process noise and measurement noise
v_list = [];
w_list = [];

for k = 1:k_bar
    % Generate truncated standard Gaussian noise v_k bounded in [-1, 1]
    v_k = max(min(randn, v_bounds(2)), v_bounds(1));
    
    % Generate Laplace distributed noise w_k ~ Lap(0, 1)
    w_k = laplace_scale * (rand < 0.5) * (-1) + laplace_scale * exprnd(1);
    
    % Update the system dynamics
    y_data(k) = C * x + w_k;  % Output measurement y_k
    x = A * x + B_v * v_k;    % State update x_{k+1}
    
    % Store the values for future use
    x_data(:, k+1) = x;
    v_list = [v_list, v_k];
    w_list = [w_list, w_k];
end

% The last measurement noise w_{k_bar}
w_k = laplace_scale * (rand < 0.5) * (-1) + laplace_scale * exprnd(1);
w_list = [w_list, w_k];
y_data(k_bar+1) = C * x + w_k;

% Define optimization variables
% Variables: x0 and u0, u1, ..., u_{k_bar-1}
%            L1 norm (new variables t0,t1,...,t_{k_bar})
n_l1 = k_bar+1;
% Quadratic objective (first and second terms)
Q_x0 = eye(n);           % Quadratic term for x0
Q_u = eye(k_bar);        % Quadratic term for control inputs u
Q_l1 = zeros(n_l1, n_l1);% Quadratic term for t_k
Q = blkdiag(Q_x0, Q_u, Q_l1);  % Combine into one quadratic matrix

% Linear term to account for -\mu_0' x0 in the objective
% Combine with control input term (zeros for u, ones for l1)
f = [-mu_0; zeros(k_bar, 1); ones(n_l1, 1)];   

% Linear inequality constraints

% Auxiliary matrices for optimization
Aux1 = zeros(k_bar+1, size(A,1));        % (k_bar+1) × n
Aux2 = zeros(k_bar+1, k_bar);            % (k_bar+1) × k_bar

for i = 0:k_bar
    Aux1(i+1, :) = C * (A^i);
    for j = 0:i-1
        Aux2(i+1, j+1) = C * (A^(i-1-j)) * B_v;
    end
end

% transform L1 norm to linear inequality constraints
% x_k = A^k * x0 + A^(k-1)*u_0 + A^(k-2)*u_1 ...
% Cx_i - t_i <= y_i
%-Cx_i - t_i <=-y_i
A_l1 = [Aux1, Aux2, -eye(n_l1); -Aux1, -Aux2, -eye(n_l1)];
b_l1 = [y_data'; -y_data'];

% Control input bounds: |u_k| <= 1
Aineq = [zeros(k_bar, n), eye(k_bar),zeros(k_bar, n_l1); zeros(k_bar, n), -eye(k_bar),zeros(k_bar, n_l1)];
bineq = ones(2*k_bar, 1);

% Combine all constraints
A_total = [Aineq; A_l1];
b_total = [bineq; b_l1];

% Solve using quadprog
% x = quadprog(H,f,A,b,Aeq,beq,lb,ub)
%
options = optimoptions('quadprog','Display','off');
[x_sol, fval] = quadprog(Q, f, A_total, b_total, [], [], [], [], [], options);

% 解ベクトルを分解
x0_est = x_sol(1:n);                     % 推定された初期状態
u_est  = x_sol(n+1:n+k_bar);             % 推定された入力系列

% 状態推定の復元
xhat = zeros(n, k_bar+1);
xhat(:,1) = x0_est;
for k = 1:k_bar
    xhat(:,k+1) = A * xhat(:,k) + B_v * u_est(k);
end

% Plot results
figure('Name', 'Figure5.4(a)');
plot(0:k_bar, x_data(1,:), 'LineWidth', 1.5);
hold on;
plot(0:k_bar, y_data, 'g--', 'LineWidth', 1.5);
plot(0:k_bar, xhat(1,:), 'r-.', 'LineWidth', 1.5);
xlabel('Time step $k$');
legend('True state $(x_k)_1$', 'Measurements $y$', 'Estimated ${(\hat{x}_k)}_1$');
grid on;

figure('Name', 'Figure5.4(b)');
plot(0:k_bar, x_data(3,:), 'LineWidth', 1.5);
hold on;
plot(0:k_bar, xhat(3, :), 'r-.', 'LineWidth', 1.5);
xlabel('Time step $k$');
legend('True state $(x_k)_3$', 'Estimated ${(\hat{x}_k)}_3$');
grid on;

figure('Name', 'Figure5.4(c)');
stairs(0:k_bar-1, v_list', 'LineWidth', 1.5);
hold on;
stairs(0:k_bar-1, u_est, 'r-.', 'LineWidth', 1.5);
xlabel('Time step $k$');
legend('Distrubance $v_k$', 'Estimation $\hat{v}_k:=u_k$');
grid on;

figure('Name', 'Figure5.4(d)');
stairs(0:k_bar, w_list', 'LineWidth', 1.5);
hold on;
stairs(0:k_bar, y_data'-xhat(1,:)', 'r-.', 'LineWidth', 1.5);
xlabel('Time step $k$');
legend('Noise $w_k$', 'Estimation $\hat{w}_k:=y-{(\hat{x}_k)}_1$');
grid on;