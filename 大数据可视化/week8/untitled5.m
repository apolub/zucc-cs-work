[X,Y] = meshgrid(-2:0.25:2,1:20);
Z = X.*Y.*exp(-X.^2-Y.^2)
surf(X,Y,Z)
colorbar
view(45,45)
