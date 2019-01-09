imagesize = 100;

imds = imageDatastore("/Users/Tobias/Downloads/rps20180219",'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadSize = numpartitions(imds);
imds.ReadFcn = @(loc)rgb2gray(imresize(imread(loc),[imagesize,imagesize]));

% Divide into training and validation data sets
numTrainFiles = 130;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


%%

% Plot one sample from each class
paper = find(imds.Labels == 'paper', 1);
rock = find(imds.Labels == 'rock', 1);
scissor = find(imds.Labels == 'scissor', 1);
 
figure
subplot(1,3,1);
imshow(readimage(imds,paper))
subplot(1,3,2);
imshow(readimage(imds,rock))
subplot(1,3,3);
imshow(readimage(imds,scissor))


% # of samples of each label
%labelCount = countEachLabel(imds)

% Dimention of examples (500x500x3)
%img = readimage(imds,1);
%size(img)

%% CNN architecture I

layers = [
    imageInputLayer([imagesize imagesize 1])
    
    convolution2dLayer(5,32,'Padding','same','Stride',2)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,64,'Padding','same','Stride',2)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(1000)
    reluLayer
    %dropoutLayer(0.3)
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

% TONOS CNN
%{
layers = [
    imageInputLayer([imagesize imagesize 1])
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Stride',1,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Stride',1,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.5)
    
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

%}
% FOLLOWING STRUCTURE YIELDS APPROX 60% ACCURACY
%{
layers = [
    imageInputLayer([227 227 3])
    
    convolution2dLayer(50,96,'Stride',4)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2)
    
    convolution2dLayer(20,256,'Stride',1,'Padding',2)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2)
    
    convolution2dLayer(3,384,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2)
    
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
%}

%% Training and validation

  options = trainingOptions('sgdm', ...
       'InitialLearnRate',0.01, ...
       'MaxEpochs',15, ...
       'Shuffle','every-epoch', ...
       'ValidationData',imdsValidation, ...
       'ValidationFrequency',2, ...
       'Verbose',false, ...
       'Plots','training-progress', ...
       'LearnRateSchedule','none',...
       'LearnRateDropPeriod',5,...
       'LearnRateDropFactor',0.1,...
       'MiniBatchSize',128);

%   options = trainingOptions('sgdm', ...
%       'MaxEpochs',40, ...
%       'ValidationData',imdsValidation, ...
%       'ValidationFrequency',4, ...
%       'Verbose',false, ...
%       'Plots','training-progress');

% Train network
net = trainNetwork(imdsTrain,layers,options);

% Classification of validation set
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

% Estimation of acuuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)