% Author: Kenji Kashima
% Date  : 2025/06/05

clear; close all;
rng(23);  % Random seed

% Step‐size parameters
C_vals     = [0.6, 0.6, 1.0, 1.0];
alpha_vals = [1.0, 0.3, 1.0, 1.5];
numSettings = numel(C_vals);

% Two simulation lengths: (a) k_bar=100, (b) k_bar=5000
k_bar_list = [100, 5000];

for idxFig = 1:2
    k_bar = k_bar_list(idxFig);
    p_history = zeros(numSettings, k_bar+1);
    p_history(:,1) = 1;  % initial value p(0)=1 for all settings

    for s = 1:numSettings
        C = C_vals(s);
        alpha = alpha_vals(s);
        for k = 1:k_bar
            p_k      = p_history(s,k);
            z_k      = randn;                    % z_k ~ N(0,1)
            p_history(s,k+1) = p_k - C/((k)^alpha) * (p_k - z_k);
        end
    end

    % Plot all trajectories in one figure
    figure('Name', sprintf('Figure 11.4(%s)', char('a'+idxFig-1))); hold on; grid on;
    k_axis = 0:k_bar;
    colors = lines(numSettings);

    for s = 1:numSettings
        plot(k_axis, p_history(s,:), 'LineWidth', 1.5, 'Color', colors(s,:));
    end

    % Mark initial value at k=0
    scatter(0, 1, 80, 'k', 'filled', 'DisplayName', 'Initial Value');

    xlim([0, k_bar]);
    ylim([-2, 2]);
    xlabel('k', 'Interpreter','latex', 'FontSize',14);
    ylabel('$p_k$', 'Interpreter','latex', 'FontSize',14);

    % Build legend entries for the four step‐size settings
    legendEntries = cell(1, numSettings+1);
    for s = 1:numSettings
        legendEntries{s} = sprintf('$C=%.1f,\\ \\alpha=%.1f$', C_vals(s), alpha_vals(s));
    end
    legendEntries{end} = 'Initial Value';

    legend(legendEntries, 'Interpreter','latex', 'FontSize',10, 'Location','best');
    hold off;
end
