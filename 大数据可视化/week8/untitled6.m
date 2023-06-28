x=1:100;
y=-1./(-2+1./x);
subplot(2,2,2);
figure(1);plot(x,y);

x=1:100;
y=-1./(0+1./x);
subplot(2,2,2);
figure(2);plot(x,y);

x=1:100;
y=-1./(2+1./x);
subplot(2,2,2);
figure(3);plot(x,y);

x=1:100;
y=-1./(4+1./x);
subplot(2,2,2);
figure(4);plot(x,y);