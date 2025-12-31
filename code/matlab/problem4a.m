% Problem 4 – Downlink Beamforming with SINR Constraints (2 users)
% -------------------------------------------------------------------------
% We minimize the total transmit power (||w||_F^2) while ensuring each user
% sees a target SINR (gamma).  This script covers the 2-user example from
% project-topic.pdf and comments explain how the SOC constraints are built.

% reset
close all;
clear all;
clc;

% setup -------------------------------------------------------------------
NUM_ANA = 8;  % BS antennas (columns in the beamforming matrix W).
NUM_MS = 2;   % User equipments (rows in H^H).

CARR_FREQ = 2.4e9;                            % Hz. carrier frequency.
WAVE_LEN = physconst('LightSpeed')/CARR_FREQ; % Lambda (m).
ANTENNA_DIS = WAVE_LEN/2;                     % Half-wavelength spacing.

% Channel model: ideal gains pointing towards ANGLE_MS directions.
CHAN_GAIN = ones(1,NUM_MS);
ANGLE_MS = [-33,  10];   % Angles of departure (deg) for each MS.

sigma_i = sqrt(0.01)*ones(1,NUM_MS);  % Noise standard deviation per MS.
gamma_dB = 10;                        % SINR target in dB.
gamma_0 = sqrt(10^(gamma_dB/10));     % Convert to linear amplitude.

% Build channel matrix H = [h_1^T; ...; h_N^T].
h = zeros(NUM_ANA, NUM_MS);
for i=1:NUM_MS
    h(:,i) = CHAN_GAIN(i) * exp(1j*(0:NUM_ANA-1).' * 2*pi*ANTENNA_DIS*sin(ANGLE_MS(i)*pi/180)/WAVE_LEN);
end


cvx_begin
    % Decision variables
    variable w(NUM_ANA, NUM_MS) complex   % Columns are per-user beams.
    variable t                            % Epigraph of ||W||_F.
    expression second_Hw(NUM_MS,NUM_MS+1)

    % Build block matrix for SOC constraint: interference + noise components.
    H = [];
    for i=1:NUM_MS
        H = [H; h(:,i).'];
    end
    H2 = H*[w,zeros(NUM_ANA,1)];  % Append noise column (filled next).
    for i=1:NUM_MS
        H2(i,i) = 0;              % Remove desired link so only interference remains.
    end
    H2(:,NUM_MS+1) = sigma_i.';   % Last column carries noise power.
   
    % Objective: shrink transmit power (Problem 4 eq. 9).
    minimize(t);
    subject to
        % Keep w inside an SOC: ||W||_F <= t.
        norm(w,'fro') <= t;
    
        for i=1:NUM_MS
            % Align the useful signal phase to the real axis to avoid ambiguity.
            real((h(:,i).')*w(:,i)) >= 0;
            imag((h(:,i).')*w(:,i)) == 0;

            % SOC form of SINR >= gamma constraint (Boyd’s formulation):
            %   || [interference; noise] ||_2 <= (1/gamma) * desired_signal
            {H2(i,:).',(1/gamma_0)*(h(:,i).')*w(:,i)} <In> complex_lorentz(NUM_MS+1);
        end
cvx_end


% Plot the composite beampattern seen across angles.
steering_vec_plot = [];
for i=-90:1:90
    steering_vec_plot = [steering_vec_plot; exp(-1j*(0:NUM_ANA-1)*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)];
end

f =  figure;
plot(-90:1:90, 10*log10(abs(w'*steering_vec_plot.').^2));
title('Minimizing the total transmit power')
legend('MS at -33^\circ', 'MS at 10^\circ');
hold on
grid on
xlabel('Angle(degree)');
ylabel('Angle Response(db)');
xlim([-90, 90])
hold off


saveas(f, 'problem4a.png')
