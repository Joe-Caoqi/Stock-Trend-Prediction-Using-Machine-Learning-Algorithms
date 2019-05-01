clear all; close all; clc
% google

X_test  = importdata('X_test_google.mat');
X_train = importdata('X_train_google.mat');
y_test  = importdata('y_test_google.mat');
y_train = importdata('y_train_google.mat');

%% Data pre-processing
flag = 0;

if flag
clear all; clc
load googleprocesseddata.mat
google_data = table2array(googleprocesseddata);
google_data = double(google_data);

y = google_data(:,end);
% for i = 1:length(y)
%     if y(i) == 0
%         y(i) = 1;
%     end
% end

X = [];
for i = 1:120
    X = [X normalize(google_data(:,i)','range')'];
end
X = double(X);

% Partition into Test and Train
SVM_Mdl_HO = fitcsvm(X,y,'Holdout',0.1,'Standardize',true);

CompactSVMModel = SVM_Mdl_HO.Trained{1};    % Extract trained, compact classifier
testInds  = test(SVM_Mdl_HO.Partition);     % Extract the test indices
trainInds = logical(1-test(SVM_Mdl_HO.Partition));   % Extract the trian indices

X_test_google  = X(testInds,:);
y_test_google  = y(testInds,:);
X_train_google = X(trainInds,:);
y_train_google = y(trainInds,:);

save X_test_google X_test_google
save y_test_google y_test_google
save X_train_google X_train_google
save y_train_google y_train_google
end


%% Cross-validation on Model parameter

% poly
CVerr = [];
Trainerr = [];
Testerr = [];
for deg = 1:6
    deg
    
    SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
        'KernelFunction','polynomial','PolynomialOrder',deg);
    
    % CV
    CVSVMModel = crossval(SVMModel,'KFold',5);
    cv_error = kfoldLoss(CVSVMModel);
    
    % Train
    [y_pred_t,~] = predict(SVMModel,X_train);
    train_error = classification_error(y_pred_t, y_train);
    
    % Test
    [y_pred,~] = predict(SVMModel,X_test);
    test_error = classification_error(y_pred, y_test);
    
    % Error
    CVerr = [CVerr cv_error];
    Trainerr = [Trainerr train_error];
    Testerr = [Testerr test_error];
end


% rbf
CVerr = [];
Trainerr = [];
Testerr = [];
KS = [0.01 0.1 1 10 100 500 1000 10000];
KS_refine = [1 10 20 30 40 50 60 70 80 90 100];
for ks = KS
    ks
    
    SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
        'KernelFunction','rbf','KernelScale',ks);

    CVSVMModel = crossval(SVMModel,'KFold',5);
    cv_error = kfoldLoss(CVSVMModel);
    
    % Train
    [y_pred_t,~] = predict(SVMModel,X_train);
    train_error = classification_error(y_pred_t, y_train);
    
    % Test
    [y_pred,~] = predict(SVMModel,X_test);
    test_error = classification_error(y_pred, y_test);
    
    % Error
    CVerr = [CVerr cv_error];
    Trainerr = [Trainerr train_error];
    Testerr = [Testerr test_error];
end


%% Cross-validation on C (Box Constraint)

Cset = [1e-2 1e-1 1 1e1 1e2 1e3 1e4];
CVerr = [];
Trainerr = [];
Testerr = [];
for C = Cset
    C
    
    SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
        'KernelFunction','linear','BoxConstraint',C);
    
%     SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
%         'KernelFunction','polynomial','PolynomialOrder',1,'BoxConstraint',C);

%     SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
%         'KernelFunction','rbf','KernelScale',20,'BoxConstraint',C);
    
    % CV
    CVSVMModel = crossval(SVMModel,'KFold',5);
    cv_error = kfoldLoss(CVSVMModel);
    
    % Train
    [y_pred_t,~] = predict(SVMModel,X_train);
    train_error = classification_error(y_pred_t, y_train);
    
    % Test
    [y_pred,~] = predict(SVMModel,X_test);
    test_error = classification_error(y_pred, y_test);
    
    % Error
    CVerr = [CVerr cv_error];
    Trainerr = [Trainerr train_error];
    Testerr = [Testerr test_error];
end


%% Optimized model - Train and Test
% linear
C = 1;
SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
    'KernelFunction','linear','BoxConstraint',C);
[y_pred_t,~] = predict(SVMModel,X_train);
train_error_linear = classification_error(y_pred_t, y_train)
[y_pred,~] = predict(SVMModel,X_test);
test_error_linear = classification_error(y_pred, y_test)

% poly
C = 1;
SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
    'KernelFunction','polynomial','PolynomialOrder',1,'BoxConstraint',C);
[y_pred_t,~] = predict(SVMModel,X_train);
train_error_poly = classification_error(y_pred_t, y_train)
[y_pred,~] = predict(SVMModel,X_test);
test_error_poly = classification_error(y_pred, y_test)

% rbf
C = 100;
SVMModel = fitcsvm(X_train,y_train,'Standardize',true,...
    'KernelFunction','rbf','KernelScale',20,'BoxConstraint',C);
[y_pred_t,~] = predict(SVMModel,X_train);
train_error_rbf = classification_error(y_pred_t, y_train)
[y_pred,~] = predict(SVMModel,X_test);
test_error_rbf = classification_error(y_pred, y_test)


%% Plots

% poly - deg
figure()
plot(CVerr,'m-o','LineWidth',1.5)
hold on
plot(Trainerr,'r-o','LineWidth',1.5)
hold on
plot(Testerr,'b-o','LineWidth',1.5)
xlabel('polynomial degree','Interpreter','latex')
ylabel('error','Interpreter','latex')
leg = legend('cross-validation','training error','testing error');
set(leg, 'Interpreter', 'latex','FontSize',15)

% rbf - ks
figure()
semilogx(KS,CVerr,'m-o','LineWidth',1.5)
hold on
semilogx(KS,Trainerr,'r-o','LineWidth',1.5)
hold on
semilogx(KS,Testerr,'b-o','LineWidth',1.5)
xlabel('$$\frac{1}{\sqrt{\gamma}}$$','Interpreter','latex')
ylabel('error','Interpreter','latex')
leg = legend('cross-validation','training error','testing error');
set(leg, 'Interpreter', 'latex','FontSize',15)
% rbf - ks - refine
figure()
plot(KS_refine,CVerr,'m-o','LineWidth',1.5)
hold on
plot(KS_refine,Trainerr,'r-o','LineWidth',1.5)
hold on
plot(KS_refine,Testerr,'b-o','LineWidth',1.5)
xlabel('$$\frac{1}{\sqrt{\gamma}}$$','Interpreter','latex')
ylabel('error','Interpreter','latex')
leg = legend('cross-validation','training error','testing error');
set(leg, 'Interpreter', 'latex','FontSize',15)

% CV on C
figure()
semilogx(Cset,CVerr,'m-o','LineWidth',1.5)
hold on
semilogx(Cset,Trainerr,'r-o','LineWidth',1.5)
hold on
semilogx(Cset,Testerr,'b-o','LineWidth',1.5)
xlabel('C','Interpreter','latex')
ylabel('error','Interpreter','latex')
leg = legend('cross-validation','training error','testing error');
set(leg, 'Interpreter', 'latex','FontSize',15)
