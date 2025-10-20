% Author: Kenji Kashima
% Date  : 2025/09/01
% Note  : requires Control System Toolbox

rng(1);                 % np.random.seed(1)
x_max = 1/2;            % for figure scale (used inside plots)

%% 1) System and Noise Model Setup
Ts = 0.1;
[Aw, Bw, Cw] = create_noise_model(Ts);
nw = size(Aw,1);

[Ar, Br, Cr] = create_double_integrator(Ts);
nr = size(Ar,1);

% Augmented plant
A  = [ Ar,              Br*Cw ;
       zeros(nw,nr),    Aw     ];
Bu = [ Br ; zeros(nw,1) ];         % control input
Bv = [ zeros(nr,1); Bw ];          % process disturbance (colored)
C  = [ Cr, zeros(1,nw) ];          % output (position)
x_dim = size(A,1);

%% 2) LQR and Noise parameters
Q  = diag([1, 10, zeros(1,nw)]);   % stage cost
R  = 1e-4;                         % input cost (scalar)
Qf = Q;                            % terminal cost
k_bar = 300;

Rv = 1.0;                          % process noise variance for v_k
Rw = 1e-4;                         % measurement noise variance for w_k

%% 3) Initial state and noise sequences
process_noise = sqrt(Rv)*randn(k_bar,1);  % v_k ~ N(0,Rv)
obs_noise     = sqrt(Rw)*randn(k_bar,1);  % w_k ~ N(0,Rw)
mu    = zeros(x_dim,1);
Sigma = eye(x_dim);
x0    = mvnrnd(mu, Sigma, 1).' * 0.1;     % x0 ~ N(0, I)*0.1

%% 4) LQR Gains (finite horizon)
[K, ~]      = lqr_control(A,  Bu, Q, R, Qf, k_bar);               % augmented
[K_r, ~]    = lqr_control(Ar, Br, Q(1:2,1:2), R, Qf(1:2,1:2), k_bar); % reduced

% zero-pad reduced K to augmented dim
K_r_aug = cell(k_bar,1);
for k = 1:k_bar
    Ki = K_r{k};                   % 1x2
    K_r_aug{k} = [Ki, zeros(1, x_dim-numel(Ki))]; % 1 x x_dim
end

%% 5) Simulations
% LQG optimal w/ one-step prediction (colored noise)
[x_true, x_hat_LQG, u, y, Sigmas, x_check, Sigmac] = ...
    simulate_lq_control(A, Bu, Bv, C, mu, Sigma, K, k_bar, x0, 'lqg_pred', Rw, Rv, process_noise, obs_noise);

% LQG optimal w/ Kalman filtering (colored noise)
[x_true_k, x_hat_LQG_k, u_k, y_k, Sigmas_k, x_check_k, Sigmac_k] = ...
    simulate_lq_control(A, Bu, Bv, C, mu, Sigma, K, k_bar, x0, 'lqg_kalman', Rw, Rv, process_noise, obs_noise);

% LQR optimal (white noise) on reduced plant
[x_LQRm, ~, u_LQRm, ~, ~, ~, ~] = ...
    simulate_lq_control(Ar, Br, Br, Cr, mu(1:2), Sigma(1:2,1:2), K_r, k_bar, x0(1:2), 'lqr', Rw, Rv, process_noise, obs_noise);

% LQR optimal (colored noise) on augmented plant using reduced gains (mismatch)
[x_LQRmm, ~, u_LQRmm, ~, ~, ~, ~] = ...
    simulate_lq_control(A, Bu, Bv, C, mu, Sigma, K_r_aug, k_bar, x0, 'lqr', Rw, Rv, process_noise, obs_noise);

%% 6) Figures 5.3(a)–(d)
figure5_3a(x_true, x_true_k, x_LQRmm, x_LQRm, k_bar);
figure5_3b(x_true, x_hat_LQG, y, Sigmas, x_check, Sigmac, k_bar);
figure5_3c(x_true, x_hat_LQG, Sigmas, x_check, Sigmac, k_bar);
figure5_3d(x_true, x_hat_LQG, Sigmas, x_check, Sigmac, k_bar);


% ====================== LQR (finite-horizon) ======================
function [K, Sigma] = lqr_control(A, Bu, Q, R, Qf, k_bar)
% Returns cell arrays:
%   K{k}     : 1 x x_dim feedback at step k
%   Sigma{k} : x_dim x x_dim cost-to-go at step k (Sigma{k_bar+1} = Qf)

Sigma = cell(k_bar+1,1);
K     = cell(k_bar,1);

Sigma{k_bar+1} = Qf;

for k = k_bar:-1:1
    S = Sigma{k+1};
    R_tilde = R + (Bu.' * S * Bu);     % 1x1
    S_tilde = A.' * S * Bu;            % x_dim x 1
    Q_tilde = Q + A.' * S * A;         % x_dim x x_dim

    % K_k = R_tilde^{-1} S_tilde^T  (1 x x_dim)
    K{k} = (S_tilde.' / R_tilde);

    % Sigma_k = Q̃ - S̃ R̃^{-1} S̃^T
    Sigma{k} = Q_tilde - (S_tilde / R_tilde) * (S_tilde.');
end
end


% ====================== LQ / LQG Simulation ======================
function [x_true, x_hat, u, y, Sigmas, x_check, Sigmac] = ...
    simulate_lq_control(A, Bu, Bv, C, mu, Sigma0, K, k_bar, x0, mode, Rw, Rv, v, w)
% mode: 'lqr', 'lqg_kalman', 'lqg_pred'
x_dim = size(A,1);

% noise sequences
if nargin < 15 || isempty(v), v = sqrt(Rv)*randn(k_bar,1); end
if nargin < 16 || isempty(w), w = sqrt(Rw)*randn(k_bar,1); end

x_true  = zeros(x_dim, k_bar+1);
x_hat   = zeros(x_dim, k_bar+1);
Sigmas  = zeros(x_dim, x_dim, k_bar+1);
x_check = zeros(x_dim, k_bar);
Sigmac  = zeros(x_dim, x_dim, k_bar);
u       = zeros(k_bar,1);
y       = zeros(k_bar,1);

x_true(:,1) = x0(:);
x_hat(:,1)  = mu(:);
Sigma = Sigma0;

for k = 1:k_bar
    Kk = K{k};  % 1 x x_dim

    if strcmp(mode,'lqr')
        u(k) = -Kk * x_true(:,k);
    else
        % prediction-based input
        if strcmp(mode,'lqg_pred')
            u(k) = -Kk * x_hat(:,k);
        end

        % 1) measurement
        y(k) = C * x_true(:,k) + w(k);

        % 2) Kalman gain terms
        M_tilde = C * Sigma * C.' + Rw;     % scalar
        L_check = Sigma * C.';              % x_dim x 1
        H_check = L_check / M_tilde;        % x_dim x 1

        % 3) posterior state
        innov = y(k) - C * x_hat(:,k);
        x_check(:,k) = x_hat(:,k) + H_check * innov;

        % 4) posterior covariance
        Sigma_check = Sigma - H_check * L_check.';
        Sigmac(:,:,k) = Sigma_check;

        % input using corrected estimate
        if strcmp(mode,'lqg_kalman')
            u(k) = -Kk * x_check(:,k);
        end

        % 5) time update (prior for next step)
        x_hat(:,k+1) = A * x_check(:,k) + Bu * u(k);
        Sigma = A * Sigma_check * A.' + Rv * (Bv * Bv.');
    end

    % 6) true system propagation
    x_true(:,k+1) = A * x_true(:,k) + Bu * u(k) + Bv * v(k);
    Sigmas(:,:,k+1) = Sigma;
end
end


% ====================== Plot: Fig 5.3(a) ======================
function figure5_3a(x_true, x_true_k, x_LQRmm, x_LQRm, k_bar)
x_max = 1/2;
t  = 0:k_bar;

figure('Name','Figure 5.3(a)'); hold on; grid on;
plot(t, x_LQRm(1,:),  'k--', 'DisplayName','LQR (white noise)');
plot(t, x_LQRmm(1,:), 'k',   'DisplayName','LQR');
plot(t, x_true(1,:),  'b',   'DisplayName','LQG');
plot(t, x_true_k(1,:),'r',   'DisplayName','LQG Kalman');
xlabel('$k$','Interpreter','latex');
legend('Location','best','Interpreter','latex');
xlim([0, k_bar]); ylim([-x_max, x_max]);
set(gcf,'Position',[100,100,720,420]);
% saveas(gcf,'Figure5_3a.pdf');
end


% ====================== Plot: Fig 5.3(b) ======================
function figure5_3b(x_true, x_hat, y, Sigmas, ~, ~, k_bar)
t  = 0:k_bar;
tc = 0:(k_bar-1);
figure('Name','Figure 5.3(b)'); hold on; grid on;
plot(t,  x_true(1,:), 'b',  'DisplayName','Position');
plot(tc, y,           'r',  'DisplayName','Measurements');
plot(t,  x_hat(1,:),  'b--','DisplayName','Estimate');
sd = sqrt(squeeze(Sigmas(1,1,:))).';
fill([t, fliplr(t)], [x_hat(1,:)+sd, fliplr(x_hat(1,:)-sd)], ...
     [0 0 1], 'FaceAlpha',0.2, 'EdgeColor','none', 'HandleVisibility','off');
xlabel('$k$','Interpreter','latex'); legend('Location','northeast','Interpreter','latex');
xlim([0,100]); ylim([-0.2, 0.2]);
set(gcf,'Position',[120,120,720,420]);
% saveas(gcf,'Figure5_3b.pdf');
end


% ====================== Plot: Fig 5.3(c) ======================
function figure5_3c(x_true, x_hat, Sigmas, ~, ~, k_bar)
t  = 0:k_bar;
x_max = 1/2;
figure('Name','Figure 5.3(c)'); hold on; grid on;
plot(t, x_true(2,:), 'b',  'DisplayName','Velocity');
plot(t, x_hat(2,:),  'b--','DisplayName','Estimate');
sd = sqrt(squeeze(Sigmas(2,2,:))).';
fill([t, fliplr(t)], [x_hat(2,:)+sd, fliplr(x_hat(2,:)-sd)], ...
     [0 0 1], 'FaceAlpha',0.2, 'EdgeColor','none', 'HandleVisibility','off');
xlabel('$k$','Interpreter','latex'); legend('Location','best','Interpreter','latex');
xlim([0,100]); ylim([-x_max, x_max]);
set(gcf,'Position',[140,140,720,420]);
% saveas(gcf,'Figure5_3c.pdf');
end


% ====================== Plot: Fig 5.3(d) ======================
function figure5_3d(x_true, x_hat, Sigmas, ~, ~, k_bar)
t  = 0:k_bar;
figure('Name','Figure 5.3(d)'); hold on; grid on;
plot(t, x_true(3,:), 'b',  'DisplayName','Colored noise');
plot(t, x_hat(3,:),  'b--','DisplayName','Estimate');
sd = sqrt(squeeze(Sigmas(3,3,:))).';
fill([t, fliplr(t)], [x_hat(3,:)+sd, fliplr(x_hat(3,:)-sd)], ...
     [0 0 1], 'FaceAlpha',0.2, 'EdgeColor','none', 'HandleVisibility','off');
xlabel('$k$','Interpreter','latex'); legend('Location','best','Interpreter','latex');
xlim([0,100]); ylim([-2, 2]);
set(gcf,'Position',[160,160,720,420]);
% saveas(gcf,'Figure5_3d.pdf');
end


% ====================== Helpers ======================
function [A, B, C] = create_double_integrator(Ts)
% Discrete double integrator (zoh)
Ac = [0 1; 0 0];
Bc = [0; 1];
Cc = [1 0];
sysc = ss(Ac, Bc, Cc, 0);
sysd = c2d(sysc, Ts, 'zoh');
A = sysd.A; B = sysd.B; C = sysd.C;
end

function [Aw, Bw, Cw] = create_noise_model(Ts)
% Colored noise model: F(s)=1/(s+0.3), Tustin, then variance normalization
s = tf('s');
Fc = 1/(s+0.3);
Fss = ss(Fc);
Fd  = c2d(Fss, Ts, 'tustin');
Aw = Fd.A; Bw = Fd.B; Cw = Fd.C;

% Normalize Bw so that output variance amplification is 1
Pw = dlyap(Aw, Bw*Bw.');          % solve P = Aw P Aw' + Bw Bw'
var_amp = Cw * Pw * Cw';          % scalar
Bw = Bw ./ sqrt(var_amp(1,1));
end