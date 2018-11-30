>> opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', .002, ...
    'LearnRateDropPeriod', 10, ...
    'MaxEpochs',80, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ValidationData',{myXValidation,myYValidation});
    
>> layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(5,10,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,50,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,300,'Padding','same')
    batchNormalizationLayer
    reluLayer   

    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(100)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
