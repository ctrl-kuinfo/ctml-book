% Author: Kenji Kashima
% Date  : 2025/09/01

clear;close all; rng(1); % random seed

k_bar = 50;       % total steps
n_sample = 20;  % number of samples
a = 1.1;        % system matrix
alpha = -0.2;   % parameter in control law
x_max = 5;      % view range


x = ones(1,k_bar+1);          % state with v=0
x_v = ones(n_sample,k_bar+1); % state with v~N(0,1)

for k = 1:k_bar
    v = randn(n_sample,1);  % n_sample independent standard normal distributions
    u = alpha*(a+alpha)^(k-1); % control law
    x(k+1) = a*x(k) + u;
    x_v(:,k+1) =a*x_v(:,k) + u + 0.1*v;
end

% Figure 4.1(a)
figure('Name','Figure 4.1(a)'); hold on; grid on;
plot(0:k_bar,x','LineWidth',1,'color',[0,0,0]+0); 
xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('$x_k$','Fontsize',16,'Interpreter','latex')
ylim([-x_max,x_max]);
movegui('northwest')

% Figure 4.1(b)
figure('Name','Figure 4.1(b)'); hold on; grid on;
plot(0:k_bar,x_v','LineWidth',0.2,'color',[0,0,0]+0.7); 
plot(0:k_bar,x','LineWidth',1,'color',[0,0,0]+0); 
xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('$x_k$','Fontsize',16,'Interpreter','latex')
ylim([-x_max,x_max]);
movegui('northeast')




