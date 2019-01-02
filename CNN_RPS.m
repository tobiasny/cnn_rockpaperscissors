
imds = imageDatastore("/Users/Tobias/Downloads/rps20180219",'IncludeSubfolders',true,'LabelSource','foldernames');

%%

% # of samples of each label
%labelCount = countEachLabel(imds)

% Dimention of examples (500x500x3)
%img = readimage(imds,1);
%size(img)

% Divide into training and validation data sets
numTrainFiles = 120;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% Define CNN architecture
layers = [
    imageInputLayer([500 500 3])
    %{
    convolution2dLayer(100,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    %}
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

% Specify training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',15, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',2, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train network
net = trainNetwork(imdsTrain,layers,options);

% Classification of validation set
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

% Estimation of acuuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)