function err=Bpfun(x,P,T,hiddennum,P_test,T_test)
%% train and test the BP network
% x£ºthe weight and threshold of a unit
% P£ºthe input training sample
% T£ºthe output of training sample
% hiddennum£ºnumber of neurons in the hidden layer
% P_test:test sample input
% T_test:the expected output of test sample 
% err£ºnorm of the prediction error

inputnum=size(P,1);       
outputnum=size(T,1);      

net=newff(minmax(P),[hiddennum,outputnum],{'tansig','logsig'},'trainlm');

net.trainParam.epochs=1000;   % the numbers of training
net.trainParam.goal=0.01;        
LP.lr=0.1;             %Learning rate
net.trainParam.show=NaN;

w1num=inputnum*hiddennum; 
w2num=outputnum*hiddennum;
w1=x(1:w1num);   %the ini weight from input layer to hidden layer
B1=x(w1num+1:w1num+hiddennum);  %ini the threshold of hidden layer
w2=x(w1num+hiddennum+1:w1num+hiddennum+w2num); %ini the threshold from hidden to ouput
B2=x(w1num+hiddennum+w2num+1:w1num+hiddennum+w2num+outputnum); %the threshold of ouput
net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=reshape(B2,outputnum,1);

net=train(net,P,T);

Y=sim(net,P_test);
err=norm(Y-T_test);
