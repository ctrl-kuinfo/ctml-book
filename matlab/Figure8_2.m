% Author: Kenji Kashima
% Date  : 2025/09/01

rng(24);  

figure8_2a(1000);
figure8_2b(5000);
figure8_2cdef(100000,'c');
figure8_2cdef(100000,'d');
figure8_2cdef(100000,'e');
figure8_2cdef(100000,'f');


% ===================== 共通初期化 =====================
function [a, b, q0, p0, Sigma0, k_bar] = initialization(k_bar)
% ARX パラメータと初期値（Python と同じ）
a = [1.2; -0.47; 0.06];   % (z-0.5)(z-0.4)(z-0.3) に対応
b = [1.0; 2.0];           % b1,b2 （Python 側は [1.0, 2.0] と表記）
q0 = [0; 0; 0; 1; 1];     % [y_?,y_?,y_?, u_?,u_?], コメント: u0=u1=1
p0 = zeros(5,1);          % 初期パラメータは 0
Sigma0 = 1e4 * eye(numel(p0));  % 初期共分散
end


% ===================== 同定モジュール（Algorithm 3） =====================
function [a_err, b_err, TrSigma, p_est, y] = sysid_module(p_star, a_dim, q0, u, v, p0, Sigma0, alpha)
% p_star: (k_dat x p_dim) 真のパラメータ列  [a1 a2 a3 b1 b2]
% a_dim : a の次元数 (=3)
% q0    : 初期データ [y_{-1}, y_{-2}, y_{-3}, u_{-1}, u_{-2}]
% u,v   : 入力/雑音（長さ k_dat）
% p0    : 初期推定
% Sigma0: 初期共分散
% alpha : 忘却係数

[k_dat, p_dim] = size(p_star);

y = zeros(k_dat-1,1);
q = zeros(k_dat, p_dim);
q(1,:) = q0(:).';

% 真のシステム生成： y(k) = p_star(k,:)*q(k,:)' + v(k)
% ここで q(k,:) = [y_{k-1}, y_{k-2}, y_{k-3}, u_{k-1}, u_{k-2}]
for k = 1:(k_dat-1)
    y(k) = p_star(k,:) * q(k,:).' + v(k);
    q(k+1,:) = [ y(k), q(k,1:a_dim-1), u(k), q(k, a_dim+1:(p_dim-1)) ];
end

p_est   = zeros(k_dat, p_dim);
a_err   = zeros(k_dat, 1);
b_err   = zeros(k_dat, 1);
TrSigma = zeros(k_dat, 1);

p_est(1,:) = p0(:).';
Sigma = Sigma0;

% k=1 の誤差（Python 同様：最初の行の p_star と p0 で計算）
a_err(1)   = sum( (p_star(1,1:a_dim) - p0(1:a_dim).').^2 );
b_err(1)   = sum( (p_star(1,a_dim+1:end) - p0(a_dim+1:end).').^2 );
TrSigma(1) = trace(Sigma);

for k = 1:(k_dat-1)
    qk = q(k,:).';
    % Algorithm 3:
    H = (Sigma*qk) / (alpha + qk.'*Sigma*qk);             % line 1
    p_est(k+1,:) = p_est(k,:) + (H * (y(k) - p_est(k,:)*qk)).'; % line 2
    Sigma = (Sigma - (H * qk.')*Sigma) / alpha;            % line 3
    % 数値対称化（Python でもやっている簡易安定化）
    Sigma = 0.5 * (Sigma + Sigma.');
    % 誤差記録
    a_err(k+1)   = sum( (p_star(k+1,1:a_dim) - p_est(k+1,1:a_dim)).^2 );
    b_err(k+1)   = sum( (p_star(k+1,a_dim+1:end) - p_est(k+1,a_dim+1:end)).^2 );
    TrSigma(k+1) = trace(Sigma);
end
end


% ===================== Figure 8.2(a) =====================
function figure8_2a(k_bar)
if nargin < 1, k_bar = 1000; end
[a, b, q0, p0, Sigma0, k_bar] = initialization(k_bar);
a_dim = numel(a);
p_star = repmat([a; b].', k_bar, 1);  % 真のパラメータ列

figure('Name','Figure 8.2(a)'); hold on; grid on;

% Case 1: u_k = 1, v_k ~ N(0,1)
sigma_v = 1.0;
v = sigma_v * randn(k_bar,1);
u = ones(k_bar,1);
[~, ~, ~, ~, y] = sysid_module(p_star, a_dim, q0, u, v, p0, Sigma0/sigma_v, 1.0);
plot(y, 'LineWidth', 0.5, 'DisplayName', '$u_k=1,\ v_k\sim\mathcal{N}(0,1)$');

% Case 2: u_k ~ N(1,1), v_k ~ N(0,0.01)
sigma_v = 0.1;
v = sigma_v * randn(k_bar,1);
u = ones(k_bar,1) + randn(k_bar,1);
[~, ~, ~, ~, y] = sysid_module(p_star, a_dim, q0, u, v, p0, Sigma0/sigma_v, 1.0);
plot(y, 'LineWidth', 0.5, 'DisplayName', '$u_k\sim\mathcal{N}(1,1),\ v_k\sim\mathcal{N}(0,0.01)$');

xlabel('$k$', 'Interpreter','latex');
ylabel('$y_k$', 'Interpreter','latex');
xlim([0, k_bar]);
legend('Location','best','Interpreter','latex');
set(gcf,'Position',[120 120 720 420]);
% grid on; box on;
% saveas(gcf, 'Figure8_2a.pdf');
end


% ===================== Figure 8.2(b) =====================
function figure8_2b(k_bar)
if nargin < 1, k_bar = 5000; end
[a, b, q0, p0, Sigma0, k_bar] = initialization(k_bar);
a_dim = numel(a);

% p^* の一部を周期的に変更（Python の mask: (np.arange(k_bar)%2000 > 1000)）
a_changed = [1.0; -0.47; 0.06];
p_star = zeros(k_bar, numel(p0));
mask = mod(0:(k_bar-1), 2000) > 1000; % 論文通りに「1001..1999」区間で切替
p_star(mask, :)  = repmat([a_changed; b].', sum(mask), 1);
p_star(~mask, :) = repmat([a;        b].', sum(~mask), 1);

sigma_v = 0.1;
v = sigma_v * randn(k_bar,1);
u = randn(k_bar,1);

figure('Name','Figure 8.2(b)'); hold on; grid on;

alpha = 1.0;
[~, ~, ~, p_est, ~] = sysid_module(p_star, a_dim, q0, u, v, p0, Sigma0/sigma_v, alpha);
plot(p_est(:,1), 'LineWidth', 0.5, 'DisplayName', '$\alpha=1$');

alpha = 0.995;
[~, ~, ~, p_est, ~] = sysid_module(p_star, a_dim, q0, u, v, p0, Sigma0/sigma_v, alpha);
plot(p_est(:,1), 'LineWidth', 0.5, 'DisplayName', '$\alpha=0.995$');

alpha = 0.8;
[~, ~, ~, p_est, ~] = sysid_module(p_star, a_dim, q0, u, v, p0, Sigma0/sigma_v, alpha);
plot(p_est(:,1), 'LineWidth', 0.5, 'DisplayName', '$\alpha=0.8$');

plot(p_star(:,1), 'k', 'DisplayName', 'True');

xlabel('$k$', 'Interpreter','latex');
ylabel('${\rm a}_1$', 'Interpreter','latex');
xlim([0, k_bar]);
ylim([0.8, 1.4]);
legend('Location','best','Interpreter','latex');
set(gcf,'Position',[140 140 720 420]);
% grid on; box on;
% saveas(gcf, 'Figure8_2b.pdf');
end


% ===================== Figure 8.2(c)(d)(e)(f) =====================
function figure8_2cdef(k_bar, label)
if nargin < 1, k_bar = 100000; end
if nargin < 2, label = 'c'; end
[a, b, q0, p0, Sigma0, k_bar] = initialization(k_bar);
a_dim = numel(a);

alpha  = 1.0;
p_star = repmat([a; b].', k_bar, 1);  % 真のパラメータ固定

% ラベルで v の分散・u の分布を切替
if label=='c' || label=='d'
    sigma_v = 1.0;
else
    sigma_v = 0.1;
end
if label=='c' || label=='e'
    v = sigma_v * randn(k_bar,1);
    u = ones(k_bar,1);
else
    v = sigma_v * randn(k_bar,1);
    u = ones(k_bar,1) + randn(k_bar,1);
end

[a_err, b_err, TrSigma, ~, ~] = sysid_module(p_star, a_dim, q0, u, v, p0, Sigma0/sigma_v, alpha);

k = (1:k_bar).';
figure('Name', ['Figure 8.2(' label ')']);
ax = axes('XScale','log','YScale','log'); hold(ax,'on'); grid(ax,'on'); grid on;
loglog(k,a_err, 'LineWidth', 1.0, 'DisplayName', '$\|\check{\rm p}^a-{\rm a}^*\|^2$');
loglog(k,b_err, 'LineWidth', 1.0, 'DisplayName', '$\|\check{\rm p}^b-{\rm b}^*\|^2$');
loglog(k,TrSigma * sigma_v, 'LineWidth', 1.0, 'DisplayName', '${\rm Trace}(\check\Sigma)$');
xlabel('$k$', 'Interpreter','latex');
xlim([1, k_bar]);
ylim([1e-10, 1e5]);
legend('Location','best','Interpreter','latex');
set(gcf,'Position',[160 160 720 420]);
% grid on; box on;
% saveas(gcf, ['Figure8_2' label '.pdf']);
end