% Problem 2(a) Variant – Average Sidelobe Energy Minimization
% -------------------------------------------------------------------------
% Same formulation as problem2.m, but with a narrower desired beam
% (15°–25°).  Comments explain how each parameter influences the beam.

% clear
clear all;
clc;
close all;

% Array / RF setup --------------------------------------------------
NUM_ANA = 40;                                 % Number of antennas.
CARR_FREQ = 2.4e9;                            % Operating frequency (Hz).
WAVE_LEN = physconst("LightSpeed")/CARR_FREQ; % Wavelength c / f.
ANTENNA_DIS = WAVE_LEN/2;                     % Half-wave spacing.



% Desired beam spans [theta_l, theta_u] degrees.
theta_l = 15;
theta_u = 25;

% Angle where we pin unit gain (center of desired window).
ANGLE_DES = (theta_l + theta_u)/2;


% Steering vector: phase shift per element for a plane wave at ANGLE_DES.
Steering_des = exp(-1j*(0:NUM_ANA-1).' * 2*pi*ANTENNA_DIS*sin(ANGLE_DES*pi/180)/WAVE_LEN);


% Final Beam Steering Vector (or Array Response Vector)
Beam_weight_MF = Steering_des/NUM_ANA;

% Covariance P integrates sidelobe steering vectors (Problem 2 objective).
P_matrix = zeros(NUM_ANA, NUM_ANA);

% interferer from -90 to theta_l degrees
for i=-90:1:theta_l
    steering_i = exp(-1j*(0:NUM_ANA-1)'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN);
    P_matrix = P_matrix + steering_i * steering_i';
end


for i= theta_u:1:90
    steering_i = exp(-1j*(0:NUM_ANA-1)'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN);
    P_matrix = P_matrix + steering_i * steering_i';
end
% Solve: min w^H P w  s.t.  w^H a(theta0) = 1
cvx_begin
    variable w(NUM_ANA,1) complex
    expression Match_output;
    Match_output = quad_form(w,P_matrix);
    minimize(Match_output);
    subject to
        w'*Steering_des == 1;
cvx_end

% Evaluate beampattern over the full angular sweep.
steering_vec_plot=[];
for i=-90:1:90
    steering_vec_plot=[steering_vec_plot; exp(-1j*(0:NUM_ANA-1)*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)];
end




% Plotter
figure(1)

plot(-90:1:90,10*log10(abs(w'*steering_vec_plot.').^2));
title('Minimizing the Average Sidelobe')

fmt = 'M=40, \theta_{d}=20, \theta_{1}=1, \theta_{u}=30';
xlabel('Angle(degree)');
ylabel('Angle Response(db)');

hold on
grid on
xlim([-90 90])
ylim([-120, 0])

% save the figure
saveas(gcf, 'problem2_1.png')
savefig(gcf, 'problem2_1.fig')
