
imds = imageDatastore("/Users/Tobias/Downloads/rps20180219",'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadSize = numpartitions(imds);
imds.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

% Divide into training and validation data sets
numTrainFiles = 120;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


%%

% paper = find(imds.Labels == 'paper', 1);
% rock = find(imds.Labels == 'rock', 1);
% scissor = find(imds.Labels == 'scissor', 1);
% 
% figure
% subplot(1,3,1);
% imshow(readimage(imds,paper))
% subplot(1,3,2);
% imshow(readimage(imds,rock))
% subplot(1,3,3);
% imshow(readimage(imds,scissor))


% # of samples of each label
%labelCount = countEachLabel(imds)

% Dimention of examples (500x500x3)
%img = readimage(imds,1);
%size(img)

%%

%convnet = alexnet;
%layers = convnet.Layers;


% Define CNN architecture
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

% Specify training options
% options = trainingOptions('sgdm', ...
%      'InitialLearnRate',0.1, ...
%      'MaxEpochs',10, ...
%      'Shuffle','every-epoch', ...
%      'ValidationData',imdsValidation, ...
%      'ValidationFrequency',4, ...
%      'Verbose',false, ...
%      'Plots','training-progress');

 options = trainingOptions('sgdm', ...
     'MaxEpochs',8, ...
     'ValidationData',imdsValidation, ...
     'ValidationFrequency',30, ...
     'Verbose',false, ...
     'Plots','training-progress');

% Train network
net = trainNetwork(imdsTrain,layers,options);

% Classification of validation set
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

% Estimation of acuuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)