% Problem 4 – Downlink Beamforming with SINR Constraints (4 users)
% -------------------------------------------------------------------------
% Same convex formulation as problem4a.m, but now 4 transmit antennas and
% 4 users located at different angles.  Extra comments highlight how the
% constraints scale with user count.

% reset
close all;
clear all;
clc;

% setup -------------------------------------------------------------
NUM_ANA = 4; % Transmit antennas.
NUM_MS = 4;  % Users to serve simultaneously.

CARR_FREQ = 2.4e9;                             % Hz.
WAVE_LEN = physconst('LightSpeed')/CARR_FREQ;  % meters.
ANTENNA_DIS = WAVE_LEN/2;                      % Uniform spacing.

% Simple channel model: each user sees a steering vector at its angle.
CHAN_GAIN = ones(1,NUM_MS);
ANGLE_MS = [-33, -78, 10, 64];   % Angles of departure for each user.

sigma_i = sqrt(0.01)*ones(1,NUM_MS);  % Noise variance per receiver.
gamma_dB = 10;
gamma_0 = sqrt(10^(gamma_dB/10));     % Linear amplitude target.

h = zeros(NUM_ANA, NUM_MS);
for i=1:NUM_MS
    h(:,i) = CHAN_GAIN(i)*exp(1j*(0:NUM_ANA-1).' * 2*pi*ANTENNA_DIS*sin(ANGLE_MS(i)*pi/180)/WAVE_LEN);
end


cvx_begin
    % Variables: w(:,k) is the beam for user k, t upper-bounds ||W||_F.
    variable w(NUM_ANA, NUM_MS) complex
    variable t
    expression second_Hw(NUM_MS,NUM_MS+1)

    % Build block matrix capturing multiuser interference + noise.
    H = [];
    for i=1:NUM_MS
        H = [H; h(:,i).'];
    end
    H2 = H*[w,zeros(NUM_ANA,1)];
    for i=1:NUM_MS
        H2(i,i) = 0;           % Remove desired signal path (only interference remains).
    end
    H2(:,NUM_MS+1) = sigma_i.';  % Last col = noise.
   
    %  objective function 
    minimize(t);
    subject to
        % constraint 1: total power <= t.
        norm(w,'fro') <= t;
    
        for i=1:NUM_MS
            % constraint 2: fix phase of desired signal.
            real((h(:,i).')*w(:,i)) >=0;
            imag((h(:,i).')*w(:,i)) == 0;

            % constraint 3: SOC SINR constraint.
            {H2(i,:).',(1/gamma_0)*(h(:,i).')*w(:,i)} <In> complex_lorentz(NUM_MS+1);
        end
cvx_end


% plot angle spectrum
steering_vec_plot=[];
for i=-90:1:90
    steering_vec_plot=[steering_vec_plot; exp(-1j*[0:NUM_ANA-1]*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)];
end

f=  figure;
plot( [-90:1:90], 10*log10(abs(w'*steering_vec_plot.').^2));
title('Minimizing the total transmit power')
legend('MS at -33^\circ', 'MS at -78^\circ', 'MS at 10^\circ', 'MS at 64^\circ', 'Location', 'southeast');
hold on
grid on
xlabel('Angle(degree)');
ylabel('Angle Response(db)');
xlim([-90, 90])
hold off


saveas(f, 'problem4b.png')

