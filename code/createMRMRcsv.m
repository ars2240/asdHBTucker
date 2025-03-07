load('asdHBTucker_gam0.1.mat'); %load tensor
% phi1 = csvread('gvLDA_40.csv',1,1);
% phi2 = csvread('pwLDA_40.csv',1,1);
% phi = [phi1, phi2];
% phi = csvread('asdHBTucker_100_mRMR.csv',1,1);

%run only once, keep constant
%or use seed
pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=size(phi,1); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets
%save('cvInd.mat','ind'); %save indices
%load('cvInd.mat'); %load indices

phiMat=tenmat(phi,1); %flatten tensor to matrix
phiMat=phiMat(:,:); %convert to matrix
phiMat=phiMat(:,sum(phiMat,1)>0); %remove columns of all zeros
%phiMat=zscore(phiMat); %normalize
asd=logical(repmat([1;0],nPat/2,1)); %binary whether or not patient has ASD

%split data based on index into training and testing sets
trainPhi=phiMat(ind,:);
trainASD=asd(ind);
testPhi=phiMat(~ind,:);
testASD=asd(~ind);

phiTr=[trainASD,trainPhi]; % combine classes with features
csvwrite('asdHBTucker_gam0.1_tr.csv',phiTr); % write csv
