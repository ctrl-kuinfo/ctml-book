% Author: Kenji Kashima
% Date  : 2025/05/22

clear;close all; rng(1); % random seed

N_iter = 10;
beta = 0.95;          %%% discount rate (LQR for 1)

% x_dim = 100; u_dim = 5;
% A = rand(x_dim,x_dim);
% A = A/max(abs(eig(A)));
% B = rand(x_dim,u_dim);
A = [ 0.8    0.9    0.86;
      0.3    0.25    1;
      0.1    0.55    0.5];  %%% A-matrix
B = [1; 0; 0];
x_dim=3; u_dim=1;

% QRS = rand(x_dim+u_dim,x_dim+u_dim);    % initialized randomly
QRS = eye(4,4);
QRS = QRS*QRS';                 % to make QRS positive definite
Q = QRS(1:x_dim,1:x_dim);           % stage cost x'Qx + u'Ru + 2*u'Sx
R = QRS(x_dim+1:x_dim+u_dim,x_dim+1:x_dim+u_dim);    
S = QRS(x_dim+1:x_dim+u_dim,1:x_dim);

% Optimal gain 
[~,P_opt,~] = dlqr(A*sqrt(beta),B*sqrt(beta),Q,R,S');

%% Value iteration %%
err_list_VI = zeros(1,N_iter);
PI = zeros(x_dim,x_dim);

i = 1;
while 1
    Rt = R + beta * B'*PI*B;
    St = S + beta * B'*PI*A;
    Qt = Q + beta * A'*PI*A; 
    PIt = Qt - St' * (Rt \ St);   %Rt\St means inv(Rt) * St
    Kt = Rt \ St; 
    if sqrt(beta)*max(abs(eig(A-B*Kt)))<1
      disp(i)
      break
    end
    i = i+1;
    PI = PIt;
end
PI_ini = PIt;
K_ini = Kt;

PIt=PI_ini;
for i=1:N_iter
    err_list_VI(i)= norm(P_opt-PIt);
    Rt = R + beta * B'*PI*B;
    St = S + beta * B'*PI*A;
    Qt = Q + beta * A'*PI*A; 
    PIt = Qt - St' * (Rt \ St);   %Rt\St means inv(Rt) * St
    PI = PIt;
end


%% Policy iteration %%
err_list_PI = zeros(1,N_iter);
Kt = K_ini;
PIt = PI_ini;
for i=1:N_iter
    err_list_PI(i) =  norm(P_opt-PIt);
    PI_Q = dlyap(sqrt(beta)*(A-B*Kt)', [eye(x_dim,x_dim) -Kt']*[Q S';S R]*[eye(x_dim,x_dim);-Kt]); 
    Rt = R + beta * B'*PI_Q*B;
    St = S + beta * B'*PI_Q*A;
    Qt = Q + beta * A'*PI_Q*A;
    PIt = Qt - St' * (Rt \ St); % for Riccati error calculation
    Kt = Rt \ St;
end

figure('Name','Figure 6.1'); hold on; grid on;
plot(0:1:N_iter-1,err_list_VI,'b')
plot(0:1:N_iter-1,err_list_PI,'r')
legend('Value Iteration','Policy Iteration','Fontsize',16)
ylim([1e-9,1000])
set(gca, "YScale","log")
