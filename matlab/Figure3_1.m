% Author: Kenji Kashima
% Date  : 2025/04/01

clear; close all;
rng(1);  % keep original MATLAB seed

set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

%% Parameters
n_tmp    = 1000;
n_x      = 2 * n_tmp + 1;   % number of x grid points
x        = linspace(-1, 1, n_x)';
dx       = x(2) - x(1);
n_k_a    = 9;               % number of time steps for Figure 3.1(a)
n_k_b    = 50;              % total steps for Figure 3.1(b)
n_sample = 10;              % number of sample trajectories

% Build transition probability matrix P (approximate uniform noise over interval)
P = zeros(n_x, n_x);
for i = 1:n_x
    xi    = x(i);
    fb    = xi + 0.1*(xi - xi^3);          % fb(x)
    noise = 0.5*(1 - abs(xi));             % c(x)
    upper = fb + noise;
    lower = fb - noise;
    % Python: idx_u = int(floor(upper*n_tmp + 0.5)) + n_tmp
    idx_u = floor(upper * n_tmp + 0.5) + n_tmp + 1;
    % Python: idx_l = int(ceil(lower*n_tmp - 0.5)) + n_tmp
    idx_l = ceil(lower * n_tmp - 0.5) + n_tmp + 1;
    for j = idx_l:(idx_u)
        P(j,i) = P(j,i) + 1.0 / (idx_u - idx_l + 1);
    end
end

%% Figure 3.1(a): propagate a uniform init on [-0.5,0.5]
% define and normalize initial distribution
init = double((x >= -0.5) & (x <= 0.5));
init = init / (sum(init) * dx);

% propagate
phi = zeros(n_x, n_k_a+1);
phi(:,1) = init;
for k = 1:n_k_a
    phi(:,k+1) = P * phi(:,k);
end

% plot
figure('Name','Figure 3.1(a)'); hold on; view(26, -107); grid on;
for k = 0:n_k_a
    plot3(...
      k*ones(n_x,1), ...
      x, ...
      phi(:,k+1), ...
      'LineWidth', 2 ...
    );
end
xlabel('$k$','Interpreter','latex','FontSize',18);
ylabel('$x$','Interpreter','latex','FontSize',18);
zlabel('$\varphi_{x_k}$','Interpreter','latex','FontSize',18);
yticks([-1 -0.5 0 0.5 1]);
movegui('northeast');

%% Figure 3.1(b): sample trajectories with matching noise scale
figure('Name','Figure 3.1(b)'); hold on; grid on;
for s = 1:n_sample
    xx = zeros(1, n_k_b);
    xx(1) = rand - 0.5;  % in [-0.5, 0.5]
    for k = 1:(n_k_b-1)
        noise = rand - 0.5;  % uniform in [-0.5,0.5]
        xi    = xx(k);
        xx(k+1) = xi + 0.1*(xi - xi^3) + noise*(1 - abs(xi));
    end
    plot(0:(n_k_b-1), xx, 'LineWidth', 2);
end
xlim([0, n_k_b-1]);
xlabel('$k$','Interpreter','latex','FontSize',18);
ylabel('$x_k$','Interpreter','latex','FontSize',18);
movegui('northwest');
