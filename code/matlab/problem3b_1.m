% Overlay the problem 3(a)/3(b) beams plus the Problem 2 baseline.
% Lets us visually compare average vs worst-case designs and aperture size.

close all;
clear all;
clc;

%f=figure;


[x1, y1] = loadfromfig('problem3a.fig');
[x2, y2] = loadfromfig('problem3b.fig');
[x3, y3] = loadfromfig('problem2.fig');

xx = [x1 x2 x3];
yy = [y1 y2 y3];


f = figure;
plot(x1, y1, 'b')
hold on
plot(x2, y2, 'r')
plot(x3, y3, 'm')

grid on
xlim([-90 90])
ylim([-140, 0])

xlabel('Normal Angle (deg)')
ylabel('Array Response (dB)')

legend('20 antenna Epigraph (problem-3a)', ...
    '40 antenna Epigraph (problem-3b)', ...
    '40 antenna (problem-2a)', ...
    'Location', 'southeast')
hold off

saveas(f, 'problem3b_1.png')


function [x1_, y1_] = loadfromfig(filename)
    % Helper: open a .fig, read its line plot data, then close it again.
    f1 = openfig(filename);
    lineHandle = findobj(f1, 'Type', 'line');
    
    x1_ = get(lineHandle, 'XData');
    y1_ = get(lineHandle, 'YData');

    close(f1);
end
