% Author: Kenji Kashima
% Date  : 2025/09/11

rng(1)
k_bar = 200;
n_sample = 5;
x_dim = 5;
A = rand(x_dim,x_dim);
A = A/max(abs(eig(A)))*0.95;
B = rand(x_dim,1);
X = dlyap(A,B*B');

figure('Position', [100 100 1500 400]);
hold on;grid on;
for s = 1:n_sample
    x_list{s} = mvnrnd(zeros(x_dim,1),X)';
    for t = 1:k_bar
        x_list{s} = [x_list{s} A*x_list{s}(:,t)+B*randn];
    end
    plot3(-k_bar/2:k_bar/2, x_list{s}(1, :), x_list{s}(2, :), 'LineWidth', 1.5);
end
ylim([-5,5])
zlim([-5,5])
yticks([-5 0 5])
zticks([-5 0 5])

fontname("Times New Roman")
fontsize(20,"points")
xlabel('$k$', 'Interpreter', 'latex', 'Fontsize', 24);
ylabel('$({\rm y})_1$', 'Interpreter', 'latex', 'Fontsize', 24);
zlabel('$({\rm y})_2$', 'Interpreter', 'latex', 'Fontsize', 24);
view([170, 15]) 
