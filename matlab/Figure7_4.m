% Author: Kenji Kashima
% Date  : 2025/09/11
% Note  : This script uses Optimization Toolbox (quadprog).

rng(1);

n_x = 100;                          % n_x - number of x-grid for plot
x_p = linspace(0,1,n_x);            % x grid points for plot

% Feature mapping size
n_f = numel(phi(0));                % number of features

% matrix phi for plot
Phi_p = zeros(n_f, n_x);
for i = 1:n_x
    Phi_p(:, i) = phi(x_p(i));
end

% true function (for reference curve)
f_real = arrayfun(@(x) f_true(x), x_p);

% weight of regularization (= observation noise covariance) 
sigma_sq = 0.01;

% number of data points
s_bar = 30;

figure7_4(sigma_sq, s_bar, x_p, Phi_p, f_real, n_f);

% ------------------------ helpers ------------------------

function v = phi(x)
    % Feature mapping
    % x - input
    v = [1; x; x.^2; x.^3; x.^4; x.^5; x.^6; x.^7; x.^8; x.^9];
end

function y = f_true(x)
    % x - input
    y = 2*sin(5*x);
end

% ------------------------ main figure ------------------------

function figure7_4(sigma_sq, s_bar, x_p, Phi_p, f_real, n_f)
    % sigma_sq - squared SD
    % s_bar    - number of data points

    % generate data from U(0,1)（等間隔）
    x = linspace(0,1,s_bar);                       % x for data
    Phi = zeros(n_f, s_bar);                       % matrix phi in (7.10)
    for s = 1:s_bar
        Phi(:, s) = phi(x(s));
    end
    y = arrayfun(@(xs) f_true(xs), x)' + randn(s_bar,1);  % noisy observation (column)

    % Design matrix A and vector b for y ≈ (para^T Phi)^T = Phi^T para
    % 目的関数は ||Phi^T * para - y||^2 / s_bar + 正則化
    A = Phi';                    % (s_bar × n_f)
    b = y;                       % (s_bar × 1)

    % quadprog オプション
    opts = optimoptions('quadprog','Display','off');

    % ---------------- optimization 1: Naive (Least Squares) ----------------
    % min ||A*para - b||^2  <=>  min 1/2 para' (2 A'A) para + (-2 A'b)' para
    H_ls = 2*(A.'*A);
    f_ls = -2*(A.'*b);
    para_naive = quadprog(H_ls, f_ls, [], [], [], [], [], [], [], opts);

    % ---------------- optimization 2: Lasso ----------------
    % min ||A x - b||^2 + sigma_sq * ||x||_1
    % 変数分割 x = u - v,  u>=0, v>=0
    % 目的：(u-v)'A'A(u-v) - 2 b'A(u-v) + sigma*(1'u + 1'v)
    % → z = [u; v] に対して
    % H = 2 [Q, -Q; -Q, Q],  f = [ -2*A'*b + sigma*1;  2*A'*b + sigma*1 ]
    Q   = A.'*A;                   % (n_f × n_f)
    c   = A.'*b;                   % (n_f × 1)

    H11 = 2*Q;  H12 = -2*Q;
    H21 = -2*Q; H22 = 2*Q;
    H_lasso = [H11, H12; H21, H22];

    f1 = -2*c + sigma_sq*ones(n_f,1);
    f2 = 2*c + sigma_sq*ones(n_f,1);
    f_lasso = [f1; f2];

    lb = zeros(2*n_f,1);  ub = [];   % u>=0, v>=0
    z0 = zeros(2*n_f,1);
    z = quadprog(H_lasso, f_lasso, [], [], [], [], lb, ub, z0, opts);
    u = z(1:n_f);  v = z(n_f+1:end);
    para_lasso = u - v;

    % ---------------- optimization 3: Ridge ----------------
    % min ||A x - b||^2 + sigma_sq * ||x||^2
    % 1/2 x'( 2 A'A + 2*sigma I ) x  +  ( -2 A'b )' x
    H_ridge = 2*(A.'*A) + 2*sigma_sq*eye(n_f);
    f_ridge = -2*(A.'*b);
    para_ridge = quadprog(H_ridge, f_ridge, [], [], [], [], [], [], [], opts);

    NAIVE_dat = (para_naive.' * Phi_p).';
    RIDGE_dat = (para_ridge.' * Phi_p).';
    LASSO_dat = (para_lasso.' * Phi_p).';

    % ---------------- Figure 7.4(a) ----------------
    figure('Units','inches','Position',[1 1 8 6]); hold on;
    scatter(x, y, 36, 'filled', 'DisplayName', '${\rm y}_s$');
    plot(x_p, NAIVE_dat, 'k', 'LineWidth',2, 'DisplayName','Least Square');
    plot(x_p, RIDGE_dat, 'b', 'LineWidth',2, 'DisplayName','Ridge');
    plot(x_p, LASSO_dat, 'r', 'LineWidth',2, 'DisplayName','Lasso');
    plot(x_p, f_real, "-.", 'LineWidth',2, 'DisplayName', '$2\sin(5{\rm x}_s)$');

    axis([0,1,-5,5]);
    xlabel('$\rm x$', 'Interpreter','latex');
    ylabel('$f({\rm x})$', 'Interpreter','latex');
    legend('Interpreter','latex');
    grid on;
    set(gca,'FontSize',12);

    % ---------------- Figure 7.4(b) ----------------
    figure('Units','inches','Position',[1 1 8 6]); hold on;
    scatter(1:n_f, abs(para_ridge), 60, 'x',  'DisplayName','Ridge','LineWidth',1.2);
    scatter(1:n_f, abs(para_lasso), 60, 'o',  'DisplayName','Lasso','LineWidth',1.2);
    xlabel('$i$', 'Interpreter','latex');
    xlim([1, n_f]);
    set(gca,'XTick',1:n_f);
    legend('Interpreter','latex');
    grid on;
    set(gca,'FontSize',12);
end