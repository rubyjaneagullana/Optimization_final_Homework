% Problem 2 – Average Sidelobe Energy Minimization (project-topic.pdf)
% -------------------------------------------------------------------------
% Goal: choose beamformer weights that keep the beam bright at theta0 while
%                             H
% shrinking the average sidelobe energy w  P w outside the look window.
% Everything below explains how each parameter feeds the CVX problem.

% clear
clear all;
clc;
close all;

% Array / RF setup --------------------------------------------------
NUM_ANA = 40;          % Number of ULA elements (more antennas -> narrower mainlobe).
CARR_FREQ = 2.4e9;     % Carrier frequency in Hz.
WAVE_LEN = physconst("LightSpeed")/CARR_FREQ;  % Wavelength lambda = c / f.
ANTENNA_DIS = WAVE_LEN/2;  % Half-wavelength spacing avoids grating lobes.

% Define the desired mainlobe sector: theta_l <= theta <= theta_u (degrees).
theta_l = 10;
theta_u = 30;

% The steering vector is normalized pointing at the middle of the window.
% This is the direction where we enforce unit response.
ANGLE_DES = (theta_l + theta_u)/2;

% Steering vector a(theta0).  Each element is e^{-j k d sin(theta)} for a ULA.
Steering_des =  exp(-1j*(0:NUM_ANA-1).' * 2*pi*ANTENNA_DIS*sin(ANGLE_DES*pi/180)/WAVE_LEN);

% Matched-filter weights (not optimal, but useful for sanity comparison).
Beam_weight_MF = Steering_des/NUM_ANA;

% Project-topic def of P: average array covariance over sidelobe sector.
% We approximate it by summing steering vectors E[a(theta)a(theta)^H] over
% all interfering directions (every 1 degree in the stop bands).
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

% CVX formulation of Problem 2 -------------------------------------
% minimize w^H P w subject to w^H a(theta0) = 1 (unity gain towards ANGLE_DES).
cvx_begin
    variable w(NUM_ANA,1) complex
    expression Match_output;
    Match_output = quad_form(w,P_matrix);  % CVX helper for w^H Pw.
    minimize(Match_output);
    subject to
        w'*Steering_des == 1;
cvx_end

% Build steering vectors for the full angular sweep (for plotting).
steering_vec_plot=[];
for i=-90:1:90
    steering_vec_plot=[steering_vec_plot; exp(-1j*(0:NUM_ANA-1)*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)];
end

% Plot the spatial spectrum on a dB scale.
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
saveas(gcf, 'problem2.png')
savefig(gcf, 'problem2.fig')
