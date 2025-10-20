% Author: Kenji Kashima
% Date  : 2025/09/11

rng(24);  

n_x = 100;                           % n_x - number of x-grid for plot
x_p = linspace(0,1,n_x);             % x grid points for plot

% Feature mapping size
n_f = numel(phi(0));                 % number of features

% matrix phi for plot
Phi_p = zeros(n_f, n_x);
for i = 1:n_x
    Phi_p(:, i) = phi(x_p(i));
end

% Draw figures
figure7_1a(20, x_p, Phi_p, n_f);
sigma_sq = [0.5, 100, 10^(-6) ];     % standard deviation sigma = 0.5, 10, 10^{-3}
figure7_1b(sigma_sq, 20, 20, x_p, Phi_p, n_f);


% ======================= Functions =======================

function v = phi(x)
    % Feature mapping
    % x - input
    v = [1; x; x.^2; x.^3; x.^4; x.^5; x.^6; x.^7; x.^8; x.^9];
end

function y = f_true(x)
    % x - input
    y = 2*sin(5*x);
end

function figure7_1a(n_sample, x_p, Phi_p, n_f)
    % n_sample - number of sample functions

    figure();
    plot(x_p, zeros(size(x_p)), 'LineWidth', 3); hold on;

    for k = 1:n_sample
        para = randn(n_f, 1);
        fx = (para.' * Phi_p);                  % 1×n_x
        plot(x_p, fx, 'LineWidth', 0.5, 'Color', [0.7,0.7,0.7]);
    end

    xlabel('$\rm x$','Interpreter','latex');
    ylabel('$f({\rm x})$','Interpreter','latex');
    xlim([0 1]); grid on;
    set(gca,'FontSize',12);
end

function figure7_1b(sigma_sq, n_sample, s_bar, x_p, Phi_p, n_f)
    % sigma_sq - list of squared SD
    % n_sample - number of sample functions
    % s_bar - number of data

    x = linspace(0,1,s_bar);                          % x for data
    Phi = zeros(n_f, s_bar);                           % matrix phi in (7.10)
    for s = 1:s_bar
        Phi(:, s) = phi(x(s));
    end
    y = arrayfun(@(xs) f_true(xs), x)' + randn(s_bar,1); % noisy observation (column)

    size_sigma_sq = numel(sigma_sq);
    mean = zeros(n_f, size_sigma_sq);
    cov  = zeros(n_f, size_sigma_sq*n_f);

    % Compute mean and cov blocks
    for l = 1:size_sigma_sq
        % tmp = I - Phi * inv(Phi'Phi + sigma^2 I) * Phi'
        % (keep variable names and structure)
        A = (Phi.' * Phi) + sigma_sq(l) * eye(s_bar);
        % use matrix division instead of inv for stability
        tmp = eye(n_f) - Phi / A * Phi.';              % n_f × n_f
        cov(:, (n_f*(l-1)+1) : n_f*l) = 0.5 * (tmp + tmp.');

        % mean = Phi * inv(Phi'Phi + sigma^2 I) * y
        mean(:, l) = Phi / A * y;                      % n_f × 1
    end

    % Plot
    figure(); hold on;

    yhat1 = (mean(:,1).') * Phi_p;  % 1×n_x
    plot(x_p, yhat1, 'LineWidth', 2, 'DisplayName', '$\sigma=0.5$', 'Color',[0 0.4470 0.7410]);

    if size_sigma_sq >= 2
        yhat2 = (mean(:,2).') * Phi_p;
        plot(x_p, yhat2, 'LineWidth', 2, 'DisplayName', '$\sigma=10$');
    end
    if size_sigma_sq >= 3
        yhat3 = (mean(:,3).') * Phi_p;
        plot(x_p, yhat3, 'LineWidth', 2, 'DisplayName', '$\sigma=10^{-3}$');
    end

    % Sampling from the posterior (first block)
    Cblk = cov(:, 1:n_f);                      % n_f × n_f
    [R, ~] = chol(Cblk, 'lower');

    for k = 1:n_sample
        z = randn(n_f,1);
        para = mean(:,1) + R * z;              % multivariate normal
        fx = (para.' * Phi_p);
        plot(x_p, fx, 'LineWidth', 0.3, 'Color', [0.7,0.7,0.7], 'HandleVisibility','off');
    end

    scatter(x, y, 36, 'filled', 'DisplayName', '${\rm y}_s$');
    legend('Interpreter','latex');

    xlabel('$\rm x$','Interpreter','latex');
    ylabel('$f({\rm x})$','Interpreter','latex');
    xlim([0 1]); grid on;
    set(gca,'FontSize',12);
end