clear all
clc
syms x y(x) 
eqns = diff(y,x) == - (y.^2)/(x.^2)
S = dsolve(eqns)


