% Beamforming

% clear
clear all;
clc;
close all;

% setup
NUM_ANA= 40; % antenna number
CARR_FREQ= 2.4e9; % Hz. carrier frequency
WAVE_LEN= physconst("LightSpeed")/CARR_FREQ; % m. wavelength
ANTENNA_DIS= WAVE_LEN/2; % distance between two antennas



% the beamwidth is from theta_l to theta_u degrees
theta_l=0;
theta_u=40;

% Angle at which the beam position is maximum
ANGLE_DES= (theta_l + theta_u)/2; % at beam center 


% Steering vector along the direction of desired signal
Steering_des=  [exp(-j*[0:NUM_ANA-1].'*2*pi*ANTENNA_DIS*sin(ANGLE_DES*pi/180)/WAVE_LEN)];


% Final Beam Steering Vector (or Array Response Vector)
Beam_weight_MF= Steering_des/NUM_ANA;

%Problem No. 2
% Minimize average sidelobe 
% accumulate all the phases for the sidelobe (interferers)
P_matrix= zeros(NUM_ANA, NUM_ANA);

% interferer from -90 to theta_l degrees
for i=-90:1:theta_l
    P_matrix= P_matrix + ...
        exp(-j*[0:NUM_ANA-1]'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN) *  ...
        exp(-j*[0:NUM_ANA-1]'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)';
end


for i= theta_u:1:90
    P_matrix= P_matrix + ...
        exp(-j*[0:NUM_ANA-1]'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN) * ... 
        exp(-j*[0:NUM_ANA-1]'*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)';
end
% optimal problem
cvx_begin
    variable w(NUM_ANA,1) complex
    expression Match_output;
    Match_output=quad_form(w,P_matrix);
    minimize(Match_output);
    subject to
    w'*Steering_des==1;
cvx_end

% plot angle spectrum
steering_vec_plot=[];
for i=-90:1:90
    steering_vec_plot=[steering_vec_plot; exp(-j*[0:NUM_ANA-1]*2*pi*ANTENNA_DIS*sin(i*pi/180)/WAVE_LEN)];
end




% plot
figure(1)

plot([-90:1:90],10*log10(abs(w'*steering_vec_plot.').^2));
title('Minimizing the Average Sidelobe')

fmt = 'M=40, \theta_{d}=20, \theta_{1}=1, \theta_{u}=30';
xlabel('Angle(degree)');
ylabel('Angle Response(db)');

hold on
grid on
xlim([-90 90])
ylim([-120, 0])

% save the figure
saveas(gcf, 'problem2_2.png')
savefig(gcf, 'problem2_2.fig')