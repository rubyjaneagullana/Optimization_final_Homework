% Problem 3(c-i) – Wide mainlobe case (0°–40° window)
% -------------------------------------------------------------------------
% Same worst-case formulation as 3(a)/3(b) but the look direction spans a
% much wider angular region, so comments focus on how that changes the
% steering vector and constraints.

close all;
clc;

theta_l = 0;
theta_u = 40;
NUM_ANA = 40;
FREQ = 2.4e9;
WAVE_LEN = physconst("LightSpeed")/FREQ;
ANTENNA_DIS = WAVE_LEN/2;

% beam center (at the middle of theta_l to theta_u)
ANGLE_DES = (theta_l + theta_u)/2;

% Beam steering vector a(theta)
Steering_des = exp(-1j*(0:NUM_ANA-1).' * 2*pi*ANTENNA_DIS*sin(ANGLE_DES*pi/180)/WAVE_LEN);


% Minimize worst-case sidelobe 
P_matrix = zeros(NUM_ANA, NUM_ANA);

% optimal problem
cvx_begin
    variable t
    variable w(NUM_ANA,1) complex
    expression Match_output
    minimize(t);
    subject to
        for i=-90:1:theta_l
            P_matrix = exp(-1j*(0:NUM_ANA-1)' *2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN) * ...
                       exp(-1j*(0:NUM_ANA-1)'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)';
            match_output = quad_form(w, P_matrix);
            match_output <= t;
        end
        for i=theta_u:1:90
            P_matrix = exp(-1j*(0:NUM_ANA-1)'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN) * ...
                       exp(-1j*(0:NUM_ANA-1)'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)';
            match_output = quad_form(w, P_matrix);
            match_output <= t;
        end

        w'*Steering_des == 1;
cvx_end

% plot angle spectrum
steering_vec_plot=[];
for i=-90:1:90
    steering_vec_plot=[steering_vec_plot; exp(-1j*(0:NUM_ANA-1)*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)];
end

% plot
f=figure;
plot(-90:1:90, 10*log10(abs(w'*steering_vec_plot.').^2));
title('Minimizing the Worst-case Sidelobe')

hold on
grid on
xlabel('Angle(degree)');
ylabel('Angle Response(db)');
xlim([-90, 90])
%ylim([-120, 0])
hold off
