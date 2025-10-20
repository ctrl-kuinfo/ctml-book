% Author: Kenji Kashima
% Date  : 2025/09/01

rng(2);

% shared parameters
k_update = 50;
Npath    = 20;

% Figure 9.3(a)
iter_Gain = 1; sigma = 2;
[x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain] = RL4LQR(sigma, iter_Gain, k_update, Npath);
plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain, '3a');

% Figure 9.3(b)
sigma = 10;
[x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain] = RL4LQR(sigma, iter_Gain, k_update, Npath);
plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain, '3b');

% Figure 9.4
iter_Gain = 5;
[x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain] = RL4LQR(sigma, iter_Gain, k_update, Npath);
plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain, '4');


% =====================================================================
function [x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain] = RL4LQR(sigma, iter_Gain, k_update, Npath)
% RL for LQR with TD+RLS (Algorithm 9.x equivalent)

beta = 0.95;

A = [0.80 0.90 0.86;
     0.30 0.25 1.00;
     0.10 0.55 0.50];
B = [1; 0; 0];

x_dim = size(A,1);
u_dim = size(B,2);
n     = x_dim + u_dim;
p_dim = n*(n+1)/2+1;

Q = eye(x_dim);
R = eye(u_dim);

% discounted optimal LQR (comparison)
[K_opt, ~, ~] = dlqr( sqrt(beta)*A, sqrt(beta)*B, Q, R );   % 1x3

K = [4.1100, 11.7519, 19.2184];          % initial gain (1x3)
x = zeros(x_dim,1);                      % initial state

x_norm_hist_p = [];
K_err_hist_p  = [];
Ups_err_hist_p= [];

for j = 1:Npath

    x_hist        = x;
    K_err_hist    = zeros(1, iter_Gain*k_update);
    Ups_err_hist  = zeros(1, iter_Gain*k_update);

    isUnstable = false;
    t_idx = 0;

    for it = 1:iter_Gain
        % --- true Pi, Ups_true for current K ---
        Ak = A - B*K;                        
        G  = [eye(x_dim); -K];
        QK = G.' * G;                        

        % Pi solves: Pi = beta*Ak*Pi*Ak' + QK
        Pi = dlyap( sqrt(beta)*Ak', QK );

        Ups_true = beta * [A, B].' * Pi * [A, B] + blkdiag(Q, R); 

        % --- TD + RLS block for k_update steps ---
        p     = zeros(p_dim,1);
        Sigma = 10*eye(p_dim);

        for k = 1:k_update
            t_idx = t_idx + 1;

            % current errors
            K_err_hist(t_idx)   = norm(K - K_opt, 'fro') / norm(K_opt, 'fro');
            Ups_est             = p_to_Ups(p, n);
            Ups_err_hist(t_idx) = norm(Ups_est - Ups_true, 'fro') / norm(Ups_true, 'fro');

            % exploration input and cost
            u    = -K*x + sigma*randn(u_dim,1);
            cost = x.'*Q*x + u.'*R*u;

            % TD update
            phi_pre = phi(x, u);             % (p_dim x 1)
            x       = A*x + B*u;             % state step
            x_hist  = [x_hist, x];

            q       = phi_pre - beta * phi(x, -K*x);

            denom   = 1 + q.'*Sigma*q;
            p       = p + (Sigma*q) * ((cost - q.'*p)/denom);
            Sigma   = Sigma - (Sigma*(q*q.')*Sigma)/denom;
            % Sigma = 0.5*(Sigma+Sigma.');   % 対称化したい場合は解除
        end

        % --- gain improvement: K = Uuu \ Uux  (eq. 9.24) ---
        Ups = p_to_Ups(p, n);                % 4x4
        Uuu = Ups(x_dim+1:end, x_dim+1:end); % 1x1
        Uux = Ups(x_dim+1:end, 1:x_dim);     % 1x3
        K   = (Uuu \ Uux);                   % 1x3 (row vector)

        % stability check
        if max(abs(eig(A - B*K))) > 1
            isUnstable = true;
            break;
        end
    end

    if ~isUnstable
        x_norm_hist_p = [x_norm_hist_p; vecnorm(x_hist,2,1)]; %#ok<AGROW>
        K_err_hist_p  = [K_err_hist_p;  K_err_hist];          %#ok<AGROW>
        Ups_err_hist_p= [Ups_err_hist_p;Ups_err_hist];        %#ok<AGROW>
    end
end
end


% =====================================================================
function phi_vec = phi(x, u)
% Feature vector for Q-function: upper-triangular vectorization of [x;u][x;u]^T
xu = [x; u];                     % n x 1
H  = xu * xu.';                  % n x n
n  = size(H,1);
phi_vec = zeros(n*(n+1)/2+1, 1);
idx = 0;
for i = 1:n
    m = n - i + 1;
    phi_vec(idx+1:idx+m) = H(i, i:end).';
    idx = idx + m;
end
phi_vec(end) = 1;   % constant term
end

% =====================================================================
function Ups = p_to_Ups(p, n)
% Recover symmetric matrix Ups from its upper-triangular vectorization
p   = p(1 : n*(n+1)/2);
Ups = zeros(n,n);
idx = 0;
for i = 1:n
    m = n - i + 1;
    Ups(i, i:end) = p(idx+1:idx+m).';
    idx = idx + m;
end
Ups = 0.5*(Ups + Ups.');
end

% =====================================================================
function plot_results(x_norm_hist_p, K_err_hist_p, Ups_err_hist_p, iter_Gain, label)
gray = [0.7 0.7 0.7];
T = size(x_norm_hist_p,2);

fig = figure('Name',['Figure 9.' label]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

% ||x|| history
nexttile; hold on; grid on;
plot(x_norm_hist_p.', 'Color', gray);
plot(mean(x_norm_hist_p,1), 'k', 'LineWidth', 2);
xlim([1, T]); yline(0,'k--','LineWidth',1);
ylabel('State norm');

% Q-function error (+ Gain error if iter_Gain>1)
nexttile; hold on; grid on;
plot(Ups_err_hist_p.', 'Color', gray);
plot(mean(Ups_err_hist_p,1), 'k', 'LineWidth', 1.5);
ylabel('Q function error'); xlim([1, T]); yline(0,'k--','LineWidth',1);

if iter_Gain > 1
    yyaxis right;
    plot(mean(K_err_hist_p,1), 'r', 'LineWidth', 2);
    ylabel('Gain error'); set(gca,'YColor',[1 0 0]);
end

xlabel('$k$', 'Interpreter','latex');
set(fig, 'Position',[120 120 700 520]);

end