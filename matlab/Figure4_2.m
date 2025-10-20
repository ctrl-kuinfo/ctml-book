% Author: Kenji Kashima
% Date  : 2025/09/01

clear;close all; rng(1); % random seed

% parameters for a continuous-time system
A = [0 4;-3 2];
B = [0;1];
C = [-0.3 -4];
csys = ss(A,B,C,0); % build CT system


T_c = 0.01; % discrete-time stepsize
dsys = c2d(csys, T_c); % build DT system from the CT system
[Ad,Bd,Cd,Dd] = ssdata(dsys); % get parameters of the DT system
                              % Note: Dd=0

k_bar = 800;   % total steps
u_max = 3.2; % view range
d = 1.0;       % quantization unit
x0 = [1;0.5];% init state


% u_k = y_k
y = zeros(1,k_bar);
x_k = x0;
for k = 1:k_bar
   y(k) = Cd * x_k;
   u_k = y(k);
   x_k = Ad * x_k + Bd * u_k;
end

subplot(1,3,1); hold on; grid on;
stairs(y,'Linewidth',1,'Color','black');
axis([0,k_bar,-u_max,u_max]);
xlabel('$k$','Interpreter','latex','Fontsize', 20);
title('$u_k=y_k$','Interpreter','latex','Fontsize', 22);
legend('$y_k$','Interpreter','latex','Fontsize', 15);
set(gca, 'FontName','Times','FontSize',14 ); 


% u_k = Q(y_k)
y = zeros(1,k_bar);
u = zeros(1,k_bar);
x_k = x0;
for k = 1:k_bar
   y(k) = Cd * x_k;
   u(k) = floor((y(k) + d/2) / d) * d;
   x_k = Ad * x_k + Bd * u(k);
end

subplot(1,3,2) ;hold on; grid on;
stairs(u,'Linewidth',1,'Color','#EDB120');
stairs(y,'Linewidth',1,'Color','black');
axis([0,k_bar,-u_max,u_max]);
xlabel('$k$','Interpreter','latex','Fontsize', 20)
title('$u_k={\mathcal Q}(y_k)$','Interpreter','latex','Fontsize', 22)
legend('$u_k$','$y_k$','Interpreter','latex','Fontsize', 15)
set(gca, 'FontName','Times','FontSize',14 ); 

% u_k = Q(y_k + z_k)
y = zeros(1,k_bar);
u = zeros(1,k_bar);
x_k = x0;
for k = 1:k_bar
   y(k) = Cd*x_k;
   z_k = rand - 0.5; % uniform distribution [-0.5,0.5]
   u(k) = floor((y(k) + z_k + d/2) / d) * d ;
   x_k = Ad * x_k + Bd * u(k);
end

subplot(1,3,3);hold on; grid on;
stairs(u,'Linewidth',1,'Color','#EDB120');
stairs(y,'Linewidth',1,'Color','black');
axis([0,k_bar,-u_max,u_max]);
xlabel('$k$','Interpreter','latex','Fontsize', 20)
title('$u_k={\mathcal Q}(y_k+z_k)$','Interpreter','latex','Fontsize', 22)
legend('$u_k$','$y_k$','Interpreter','latex','Fontsize', 15)
set(gca, 'FontName','Times','FontSize',14 ); 
