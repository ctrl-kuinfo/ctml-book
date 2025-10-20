% Author: Kenji Kashima
% Date  : 2025/04/01

clear;close all; rng(6); % random seed

% Figure 3.2(a)
k_bar = 50;      % total steps
n_sample = 9; % number of samples 
figure('Name','Figure 3.2(a)'); hold on;grid('on'); 

for i = 1:n_sample
    x = zeros(1,k_bar); 
    x(1) = -1.0+ 0.2*i; % initialization
    for k = 1:k_bar-1
        x_i = x(k);
        x(k+1) =  x_i + 0.1*(x_i-x_i^3) ; 
    end
    plot(x,LineWidth=2)
end
xlim([1,k_bar])
xlabel('$k$','Interpreter','latex','Fontsize',18);
ylabel('$x_k$','Interpreter','latex','Fontsize',18);
movegui('northwest')


% Figure 3.2(b)
k_bar = 50;      % total steps
n_sample = 10; % number of samples 
figure('Name','Figure 3.2(b)'); hold on;grid('on'); 

for i = 1:n_sample
    x = zeros(1,k_bar); 
    x(1) = -0.5; % initialization
    % Note: rand is a single uniformly distributed random 
    % number in the interval (0,1).
    for k = 1:k_bar-1
        x_i = x(k);
        x(k+1) =  x_i + 0.1*(x_i-x_i^3) + (rand-0.5)*(1-abs(x_i)); 
    end
    plot(x,LineWidth=2)
end
xlim([1,k_bar])
ylim([-1,1])
xlabel('$k$','Interpreter','latex','Fontsize',18);
ylabel('$x_k$','Interpreter','latex','Fontsize',18);
movegui('northeast')
