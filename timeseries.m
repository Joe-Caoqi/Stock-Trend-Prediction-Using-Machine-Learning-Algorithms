clear all
clc
X=importdata('/Users/xingranchen/Documents/MATLAB/CIS_520/projects/adm.txt');
%X=xlsread('/Users/xingranchen/Documents/MATLAB/CIS_520/projects/mo');
% 1-open 2-high 3-low 4-close 5-adj_close 6-volume
X_open=X(:,1);
%X_close=X(:,2);
X_close=X(:,4);
%plot(X_close);

R=X_close-X_open;
%plot(R);
%R=diff(R);

h=adftest(R);



%n=length(R);
[a,b]=autocorr(R);

[c,d]=parcorr(R);
t=1:20;
% x1=0.038.*ones(1,20);
% x2=-x1;
% x=1:20;
% stem(t,a(2:end)','r*');hold on
% plot(x,x1,'b',x,x2,'b')
% xlabel('Lag'); 
% ylabel('Sample Autocorrelation');
% 
% title('Sample Autocorrelation Function'); 

x1=0.027.*ones(1,20);
x2=-x1;
x=1:20;
stem(t,c(2:end)','r*');hold on
plot(x,x1,'b',x,x2,'b')
xlabel('Lag'); 
ylabel('Sample Partial Autocorrelation');

title('Sample Partial Autocorrelation Function'); 

% 
%   sys=armax(R,[15,15]);
%   Rp=predict(sys,R,1);
%   X_close_new=X_open+Rp;
%   Y1=diff(X_close);
%   Y2=diff(X_close_new);
%   y1=sign(Y1);
%   y2=sign(Y2);
% 
%   err1=1-length(find(y1~=y2))/length(y1)  
%   err2=norm(X_close_new-X_close)^2/length(X_close)