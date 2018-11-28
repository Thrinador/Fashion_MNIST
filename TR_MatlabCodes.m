%import Data
>> data = readtable('%<<insertpath>>%');
>> myYTrain = table2array(data(:,2));
>> myYTrain = categorical(myYTrain);
>> imageVector = table2array(data(:,3:786));
>> imageVector = imageVector';
>> imageVector = reshape(imageVector,[28 28 1 60000]);

%create test set from data
idx = randperm(size(imageVector,4),1000);
myXValidation = imageVector(:,:,:,idx);
imageVector(:,:,:,idx) = [];
myYValidation = myYTrain(idx);
myYTrain(idx) = [];

%set options for trainNetwork function
>> opts = trainingOptions('sgdm', ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{myXValidation,myYValidation});

%set layers for trainNetwork function
>> layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(4,10,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,25,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,40,'Padding','same')
    batchNormalizationLayer
    reluLayer   

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(4,100,'Padding','same')
    batchNormalizationLayer
    reluLayer   

    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%run network
net = trainNetwork(imageVector,myYTrain,layers,opts);

%-------------------------------------------------------------------------------------------------------------------
%test network
  %import data
  >> testdata = readtable('%<<insertpath>>%');
>> testArray = table2array(testdata(:,2:785));
>> testArray = testArray';
>> testArray = reshape(testArray,[28 28 10000]);
>> ids = testdata(:,1);

%-------------------------------------------------------------
%test function defintion
function testTrainedNet(net,testArray,ids)
%get output of network in categorized labels and output to xls.
%net is trained network auto layer is 'classoutput'
labels = zeros(1,size(testArray,3));
%loop through array and test each input
for i = 1:size(testArray,3)
    output = activations(net,testArray(:,:,i),'classoutput','OutputAs','rows');
    %convert output to label
    output = round(output);
    label = find(output==1);
    label = label-1;
    if isempty(label) || label == -1
        label = 0;
    end
    labels(i) = label;
end
labels = labels';
T = [ids labels];
T = array2table(T,'VariableNames', {'Id','label'});
%export to cvs
writetable(T,'test.csv')
end
%------------------------------------------------------------------
%run testing
>> testTrainedNet(net,testArray,ids);
