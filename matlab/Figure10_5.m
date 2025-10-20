% Author: Kenji Kashima
% Date  : 2024/10/17

clear;close all; rng(1); % random seed

grid_size=100;
%% define the importance V(z)
value = zeros(grid_size,grid_size);
for i = 1:grid_size
    for j = 1:grid_size
        value(i,j) = 1.5*exp(-((i-20)^2)/1000-((j-20)^2)/1000)...
        +exp(-(i-20)^2/1000-(j-90)^2/500)...
        +exp(-(i-40)^2/1000-(j-50)^2/500)...
        +exp(-(i-70)^2/300-(j-70)^2/500)...
        +1.5*exp(-(i-80)^2/300-(j-40)^2/500)...
        +exp(-(i-50)^2/800-(j-50)^2/800)...
        +1.2*exp(-(i-80)^2/200-(j-20)^2/200)...    
        +1*exp(-(i-90)^2/200-(j-10)^2/200);    
    end
end
value = value/10;

% Figure 10.5(a)
figure;
contourf(value', 20); 
colorbar; 
title('V(z)');
xlabel('z_1');
ylabel('z_2');
xlim([1,100])
ylim([1,100])

function [ini_state,state,x1_list,x2_list]=simulation(value,Tmax, deterministic)
    %% update position of sensors
    IDn = 12; 
    sensor_range = 8;
    beta=1;
    x1_list =zeros(IDn,Tmax);
    x2_list =zeros(IDn,Tmax);
    ini_state = ones(IDn,2)*10; % pos_x1, pos_x2
    state = ini_state;
    for t = 1:Tmax
        x1_list(:,t) = state(:,1);
        x2_list(:,t) = state(:,2);
        
        % estimation
        state_hat = state;
        select_ID = randi(IDn);      %randomly select a sensor
        select_direction = randi(4); %randomly select a direction
        switch select_direction
           case 1 %go right
                state_hat(select_ID,1) = min(100,state_hat(select_ID,1)+1);
           case 2 %go left
                state_hat(select_ID,1) = max(1,state_hat(select_ID,1)-1);
           case 3 %go up
                state_hat(select_ID,2) = min(100,state_hat(select_ID,2)+1);
           case 4 %go down
                state_hat(select_ID,2) = max(1,state_hat(select_ID,2)-1);
        end
    
        X1=state(select_ID,1); 
        X2=state(select_ID,2); 
    
        
        l_x = weighted_state_position_value(state,sensor_range,X1,X2,value);
        l_x_hat = weighted_state_position_value(state_hat,sensor_range,X1,X2,value);
        
        l_all = [l_x,l_x_hat];
        move = soft_max(l_all,beta,deterministic); 
    
        if move > 1
            state = state_hat;
        else
            %state = state;
        end
    end
end

% Figure 10.5(b)
[ini_state,state,x1_list,x2_list] = simulation(value,20000,false);
figure; hold on;
plot(ini_state(:,1),ini_state(:,2),'*','MarkerSize',10);
plot(state(:,1),state(:,2),'o','MarkerSize',10);
t = linspace(0,2*pi,100);
for i=1:12
    plot(x1_list(i,:),x2_list(i,:));
    plot(10*sin(t)+state(i,1),10*cos(t)+state(i,2))
end
grid on;
xlim([1,100])
ylim([1,100])

% Figure 10.5(c)
[ini_state,state,x1_list,x2_list] = simulation(value,20000,true);
figure; hold on;
plot(ini_state(:,1),ini_state(:,2),'*','MarkerSize',10);
plot(state(:,1),state(:,2),'o','MarkerSize',10);
t = linspace(0,2*pi,100);
for i=1:12
    plot(x1_list(i,:),x2_list(i,:));
    plot(10*sin(t)+state(i,1),10*cos(t)+state(i,2))
end
grid on;
xlim([1,100])
ylim([1,100])

function weighted_value = weighted_state_position_value(state,distance,X1,X2,value)
    weighted_value = 0;
    for I = max(1,X1-11):min(100,X1+11)     % sensor range = 10
        for J = max(1,X2-11):min(100,X2+11)
            weighted_value = weighted_value + value(I,J)*state_position_value(state,I,J,distance);
        end
    end
end

function value = state_position_value(state,X1,X2,distance)
    value = 0;
    IDn = size(state,1);
    
    for ID = 1:IDn
        value = min( 1, value + ( (state(ID,1)-X1)^2 + (state(ID,2)-X2)^2 < distance^2 ) );
    end
end

% equation (10.40)
function selected = soft_max(l_all,beta, deterministic)
    l_all = exp(beta*l_all);
    threshold = l_all(1) / sum(l_all);
    if deterministic == true
        if threshold < 0.5
            selected = 2;
        else
            selected = 1;
        end
    else
        if rand() > threshold
            selected = 2;
        else
            selected = 1;
        end
    end
end

