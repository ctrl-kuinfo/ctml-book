% Author: Kenji Kashima
% Date  : 2025/09/01
% Note  : requires Control System Toolbox

clear;close all; rng(1); % random seed

%% Create noise model
%% --- 1. Noise Model Normalization ---
% Continuous-time noise model F = 1/(s+0.3)
s = tf('s');
F_c = 1 / (s + 0.3);
F_ss = ss(F_c);
Ts = 0.1;
F_d = c2d(F_ss, Ts, 'tustin');
Aw = F_d.A;
Bw = F_d.B;
Cw = F_d.C;

%% --- 2. Noise Model Normalization ---
% Solve discrete Lyapunov equation: Pw = Aw * Pw * Aw' + Bw * Bw'
Pw = dlyap(Aw, Bw*Bw');
variance_amplification = Cw * Pw * Cw';   % scalar variance
Bw = Bw / sqrt(variance_amplification(1,1));


% Figure 3.4(a)
% 離散時間システム
dsys = ss(Aw, Bw, Cw, 0, Ts);

% 周波数応答（0.001 ~ π）
omega = (0.001:0.001:pi).';           % 列ベクトルにしておく
H = freqresp(dsys, omega / Ts);
mag = abs(squeeze(H)); 
figure('Name','Figure 3.4(a)'); hold on; grid on;

semilogx(omega, mag, ...
    'LineWidth', 1.0, 'Color', 'b', ...
    'DisplayName', 'frequency weight');

semilogx([0.001, 0.01, 0.6, pi], [8, 8, 0, 0], ...
    'LineWidth', 2, 'LineStyle', '--', 'Color', 'r', ...
    'DisplayName', 'prior information');

set(gca, 'XScale', 'log');   % 念のため強制
xlim([1e-3, pi]);
xlabel('$\varpi$','Fontsize',18,'Interpreter','latex');
ylabel('Magnitude','Fontsize',18,'Interpreter','latex');
legend('Fontsize',16,'Interpreter','latex');
movegui('northwest');

% Figure 3.4(b)
k_bar = 200;
x_bar_dim = size(Aw,1);
v = randn(1, k_bar+1);             % white noise
x_bar = zeros(x_bar_dim, k_bar+1); % state of weighting filter
y = zeros(1, k_bar+1);             % colored noise

for k = 1:k_bar
    x_bar(:,k+1) = Aw * x_bar(:,k) + Bw * v(k);

    y(k+1) = Cw * x_bar(:,k+1);
end

figure('Name','Figure 3.4(b)'); hold on; grid on;
stairs(v)
stairs(y,'linewidth',1)
xlim([0,200])
ylim([-2,2])
xlabel('$k$','Fontsize',18,'Interpreter','latex')
legend('white','colored','Fontsize',16)
movegui('northeast')
