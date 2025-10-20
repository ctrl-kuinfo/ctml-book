% Author: Kenji Kashima
% Date  : 2025/09/01 

seed = 1;

% ===== Figure 12.2 =====
figure12_2(100, 0.1, 'a');  % Figure12.2(a)
figure12_2(100, 1.0, 'b');  % Figure12.2(b)

% ===== Figure 12.3 (Gaussian kernel) =====
figure12_3and4(100, 10, 0.1, 'a', "Gaussian", seed);  % Figure12.3(a)
figure12_3and4(100, 50, 0.1, 'b', "Gaussian", seed);  % Figure12.3(b)

% ===== Figure 12.4 (min kernel) =====
figure12_3and4(100, 10, 0.1, 'a', "min", seed);       % Figure12.4(a)
figure12_3and4(100, 50, 0.1, 'b', "min", seed);       % Figure12.4(b)



% -------------------------------------------------------------------------
function figure12_2(N_x, c, label)

x = linspace(0,1,N_x).';                 % (N_x×1)
K = gaussian_kernel(x, x, c);            % (N_x×N_x)
prior_mean = x;                           % mu(x)=x

Ks = 0.5*(K + K.');                       % 数値的に対称化のみ
[V, D] = eig(Ks, 'vector');               % D: eigenvalues (N_x×1)
D = max(D, 0);                            % 負の微小固有値は 0 に（実数化）
S = V .* sqrt(D.');                       % 相当する「平方根」作用（N_x×N_x）
% y = prior_mean + S * randn(N_x,1);

figure('Name', ['Figure 12.2(' label ')'] ); hold on; grid on;
for i = 1:5
    y = prior_mean + S * randn(N_x,1);
    plot(x, y, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, 'HandleVisibility','off');
end
plot(x, prior_mean, 'b-', 'LineWidth', 1.2, 'DisplayName', '$\mu(x)$');

x_SD = sqrt(max(0, diag(K)));
fill([x; flipud(x)], [prior_mean+x_SD; flipud(prior_mean-x_SD)], ...
     [0 0 1], 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility','off');

xlim([0,1]); ylim([-3,3]);
xlabel('x'); legend('Location','northwest');
title(['Figure 12.2(' label, ')']);
end


% -------------------------------------------------------------------------
function figure12_3and4(N_x, s_bar, c, label, kernel, seed)

rng(seed);

x_data = rand(s_bar,1) .* rand(s_bar,1); % (s_bar×1)
x      = linspace(0,1,N_x).';            % (N_x×1)

% カーネル選択
switch string(kernel)
    case "Gaussian"
        K    = gaussian_kernel(x_data, x_data, c);  % (s_bar×s_bar)
        Ktmp = gaussian_kernel(x,      x,      c);  % (N_x×N_x)
        kx   = gaussian_kernel(x,      x_data, c);  % (N_x×s_bar)
        title_tag = ['3' label];         % Figure12.3(a)/(b)
    case "min"
        K    = min_kernel(x_data, x_data);
        Ktmp = min_kernel(x,      x);
        kx   = min_kernel(x,      x_data);
        title_tag = ['4' label];         % Figure12.4(a)/(b)
    otherwise
        error('Unknown kernel: %s', kernel);
end

% 例 12.1.9 と Fig 12.4 の関数
mu_x = @(v) v;                         % mu(x) = x
fn_x = @(v) sin(4*pi*v);               % y(x) = sin(4πx)

sigma = 0.1;                           % ノイズの標準偏差
e_s = sigma * randn(s_bar,1);          % ノイズ
y_data = fn_x(x_data) + e_s;           % 観測

figure('Name', ['Figure 12.' title_tag] ); hold on; grid on;
plot(x, mu_x(x), 'r-', 'LineWidth', 1.2, 'DisplayName', '$\mu(x)$');
plot(x, fn_x(x), 'k--', 'LineWidth', 1.2, 'DisplayName', '$sin(4\pi x)$');
scatter(x_data, y_data, 36, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none', ...
        'DisplayName', '$y_s$');

% ===== 事後平均・分散 =====
prior_mean = mu_x(x_data);                 % (s_bar×1)
A = K + sigma^2 * eye(s_bar);              % そのまま（jitter なし）

% alpha = (K + σ²I)^{-1} (y - μ) を inv を使わずに解く
alpha = A \ (y_data - prior_mean);         % (s_bar×1)

% 事後平均: μ(x) + k(x,X) * alpha
post_mean = mu_x(x) + kx * alpha;          % (N_x×1)

% 事後分散の対角: diag(Kxx) - diag(kx * A^{-1} * kx')
% A^{-1} * kx' をまとめて解く
W = A \ (kx.');                            % (s_bar×N_x)

% diag(kx * W) を効率よく計算
second_term_diag = sum(kx .* (W.'), 2);    % (N_x×1)
post_var_diag    = diag(Ktmp) - second_term_diag;
post_SD          = sqrt(max(0, post_var_diag));

% 囲み塗りつぶし ±1SD
fill([x; flipud(x)], [post_mean+post_SD; flipud(post_mean-post_SD)], ...
     [0 0 1], 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility','off' );

plot(x, post_mean, '-.', 'Color', [0 0 1], 'LineWidth', 1.2, ...
     'DisplayName', '$\mu(x|\mathcal{D})$');

xlim([0,1]); ylim([-1.5, 3.0]);
yticks([-1,0,1,2,3]);
xlabel('x'); legend('Location','northwest');
title(['Figure 12.' title_tag])
end


% -------------------------------------------------------------------------
function K = gaussian_kernel(x, y, c)
% Gaussian (RBF) kernel:
% K_ij = exp(-((x_i - y_j)^2) / c^2)
x = x(:); y = y(:);
K = exp( - ( (x - y.').^2 ) / (c^2) );
end

function K = min_kernel(x, y)
% Min kernel:
% K_ij = min(x_i, y_j)
x = x(:); y = y(:);
K = min(x, y.');
end