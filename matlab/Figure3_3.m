% Author: Kenji Kashima
% Date  : 2025/04/01

clear;close all; rng(1); % random seed

n_sample = 100; % number of samples
k_bar = 10;      % number of k

a = 0.5;       % gain
x_max = 10;    % range of y_lim

x = zeros(n_sample,k_bar+1); x_noise=zeros(n_sample,k_bar+1);
x(:,1) = randn(n_sample,1); x_noise(:,1) = randn(n_sample,1);

step = randn(n_sample,1);

for t = 1:k_bar
    noise = randn(n_sample,1);
    x(:,t+1) =  a*x(:,t) + noise*sqrt(3);
    x_noise(:,t+1) = a*x_noise(:,t) + step ;
end


% Figure 3.3(a)
figure('Name','Figure 3.3(a)');grid on;
plot(0:k_bar,x_noise','LineWidth',.5); 

xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('$x_k$','Fontsize',16,'Interpreter','latex')
ylim([-x_max,x_max]);
movegui('northeast')

% Figure 3.3(b)
figure('Name','Figure 3.3(b)');grid on;
plot(0:k_bar,x','LineWidth',.5); 

xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('$x_k$','Fontsize',16,'Interpreter','latex')
ylim([-x_max,x_max]);
movegui('northwest')
