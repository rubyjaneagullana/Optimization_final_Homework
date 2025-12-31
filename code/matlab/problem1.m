% Problem 1 – Bound-Constrained Least Squares (project-topic.pdf)
% -------------------------------------------------------------------------
% We simulate a textbook problem: find x that minimizes ||Ax - b||_2^2 while
% keeping every entry of x between lower bounds l and upper bounds u.
% CVX solves the small convex program so that we can focus on how the data
% (A, b, l, u) get built and interpreted.

% clear
close all; 
clear all;
clc;


% Random-but-repeatable problem setup -------------------------------
rng(1234);  % Fix the random seed so the same "experiment" can be rerun.

m = 4;  % Number of equations, i.e., rows in A (think: measurements).
n = 2;  % Number of unknowns, i.e., columns in A (think: parameters).

A = randn(m,n);   % The sensing / design matrix.
b = randn(m,1);   % The “observed” right-hand side.
bnds = randn(n,2);  % Temporary storage for random lower / upper limits.
l = min( bnds, [], 2 );  % Element-wise lower bounds for x.
u = max( bnds, [], 2 );  % Element-wise upper bounds for x.

% Convex program in CVX ---------------------------------------------
% expression t1 is the residual norm ||Ax - b||_2, which CVX squares
% internally when minimizing the quadratic norm. The bound constraints
% encode (2) from the homework statement.
cvx_begin
    variable x(n)
    expression t1
    t1 = norm(A*x-b);  % L2 distance between model Ax and data b.
    minimize(t1)
    subject to
        l <= x <= u
cvx_end



disp("A = ")
disp(A)

disp("b = ")
disp(b)


disp("Min bound is")
disp(l')

disp("Max bound is")
disp(u')

disp("CVX calculated x as")
disp(x')
