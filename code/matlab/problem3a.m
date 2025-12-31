% Problem 3(a) – Worst-Case Sidelobe Minimization (20 antennas)
% -------------------------------------------------------------------------
% Rather than average sidelobe energy, we now bound the worst sidelobe
% magnitude over the stop band.  CVX variable t is the upper envelope.

clear all;
clc;
close all;

% for part 3(a), set NUM_ANA=20
NUM_ANA = 20;  % Number of antennas, directly controls array aperture.

% setup
CARR_FREQ = 2.4e9;                         % Hz. carrier frequency.
WAVE_LEN = physconst("LightSpeed") / CARR_FREQ;  % Lambda = c / f.
ANTENNA_DIS = WAVE_LEN/2;                  % Spacing (half lambda).

% beam width (from theta_l to theta_u)
theta_l = 10;
theta_u = 30;

% beam center (at the middle of theta_l to theta_u)
ANGLE_DES = (theta_l + theta_u)/2;  % Desired steering angle.

% Beam steering vector a(theta)
Steering_des = exp(-1j*(0:NUM_ANA-1).' * 2*pi*ANTENNA_DIS*sin(ANGLE_DES*pi/180)/WAVE_LEN);


% Minimize worst-case sidelobe --------------------------------------
P_matrix = zeros(NUM_ANA, NUM_ANA);  % Placeholder reused in the loop.

cvx_begin
    variable t                     % Worst sidelobe energy (epigraph var).
    variable w(NUM_ANA,1) complex   % Beamformer to be designed.
    expression match_output
    minimize(t);
    subject to
        % Loop over sidelobe angles and force the quadratic form <= t.
        for i=-90:1:theta_l
            P_matrix = exp(-1j*(0:NUM_ANA-1)'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN) * ...
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

        % Preserve unity gain towards the desired look direction.
        w'*Steering_des == 1;
cvx_end

% plot angle spectrum
steering_vec_plot = [];
for i=-90:1:90
    steering_vec_plot = [steering_vec_plot; exp(-1j*(0:NUM_ANA-1)*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)];
end

% plot
f = figure;
plot(-90:1:90, 10*log10(abs(w'*steering_vec_plot.').^2));
title('Minimizing the Worst-case Sidelobe')

hold on
grid on
xlabel('Angle(degree)');
ylabel('Angle Response(db)');
xlim([-90, 90])
%ylim([-120, 0])
hold off

saveas(f, 'problem3a.png')
savefig(f, 'problem3a.fig')
