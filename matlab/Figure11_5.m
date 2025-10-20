% Author: Kenji Kashima
% Date  : 2025/06/05
% Note  : Figure 11.5(a) and 11.5(b) in one run, variable names match Python version

clear; close all;
rng(23);  % Random seed

%% Common definitions
p_vals = -5:0.01:5;

L1_sym      = (p_vals.^2) / 2 + p_vals / 2;              % L1(p) = p^2/2 + p/2
grad_L1_sym = p_vals + 1/2;                              % ∇L1(p) = p + 1/2

L2_sym      = -p_vals .* cos(10 * p_vals) / 20 + sin(10 * p_vals) / 200 - p_vals / 2;
                                                          % L2(p) = -p·cos(10p)/20 + sin(10p)/200 - p/2
grad_L2_sym = (p_vals .* sin(10 * p_vals)) / 2 - 1/2;     % ∇L2(p) = p·sin(10p)/2 − 1/2

L_sym       = (L1_sym + L2_sym) / 2;                     % L(p) = (L1(p) + L2(p)) / 2
grad_L_sym  = (grad_L1_sym + grad_L2_sym) / 2;            % ∇L(p) = (∇L1(p) + ∇L2(p)) / 2

%% Figure 11.5(a)
figure('Name','Figure 11.5(a)'); hold on; grid on;

plot(p_vals, L_sym,      'LineWidth',1);
plot(p_vals, grad_L_sym, 'LineWidth',1);
plot(p_vals, grad_L1_sym,'LineWidth',1);
plot(p_vals, grad_L2_sym,'LineWidth',1);

xlabel('$x$','Interpreter','latex','FontSize',18);
legend({ ...
    '$L(x)$', ...
    '$\nabla L(x)$', ...
    '$\nabla L_{1}(x)$', ...
    '$\nabla L_{2}(x)$' ...
    }, ...
    'Interpreter','latex','FontSize',10,'Location','best');

xlim([-5, 5]);
ylim([-5, 7]);

%% Figure 11.5(b): SGD simulation for solving L(p)=0
k_bar = 2000;
C     = [1.0, 1.0, 1.0];      % Step‐size constants
alpha = [0.4, 0.8, 1.2];      % Decay exponents
N_setting = numel(C);

p_list = zeros(N_setting, k_bar + 1);
y_list = zeros(N_setting, k_bar);

p_ini = 1;
p_list(:,1) = p_ini;

figure('Name','Figure 11.5(b)'); hold on; grid on;

for setting = 1:N_setting
    for k = 1:k_bar
        p = p_list(setting, k);
        % Randomly choose ∇L1 or ∇L2 with equal probability
        if rand() < 0.5
            y = p + 1/2;                    % grad_L1_sym
        else
            y = (p * sin(10 * p)) / 2 - 1/2; % grad_L2_sym
        end
        y_list(setting, k) = y;
        % Update: p_{k+1} = p_k − (C(setting) / k^alpha(setting)) * y
        p_list(setting, k + 1) = p - C(setting) / (k ^ alpha(setting)) * y;
    end
    plot(0:k_bar, p_list(setting, :), 'LineWidth', 1);
end

% Initial value marker
plot(0, p_ini, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', 'Initial Value');

xlabel('$k$','Interpreter','latex','FontSize',18);
ylabel('$p_{k}$','Interpreter','latex','FontSize',18);
xlim([0, k_bar]);
ylim([-2, 2]);

% Legend labels 
legend_labels = cell(1, N_setting + 1);
for idx = 1:N_setting
    legend_labels{idx} = sprintf('$C=%.1f,\\ \\alpha=%.1f$', C(idx), alpha(idx));
end
legend_labels{end} = 'Initial Value';

legend(legend_labels, 'Interpreter','latex','FontSize',10,'Location','best');
