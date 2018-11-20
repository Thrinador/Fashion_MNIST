function t = testCode(td, t)
    inputLayer = imageInputLayer([784 1]);
    c1 = convolution2dLayer([200 1],20,'stride',1);
    p1 = maxPooling2dLayer([28 1],'stride',10);
    %c1 = convolution2dLayer([200 1],20);
    %p1 = maxPooling2dLayer([28 1]);
    c2 = convolution2dLayer([30 1], 400);
    p2 = maxPooling2dLayer([10 1],'stride',[1 2]);
    f1 = fullyConnectedLayer(500);
    f2 = fullyConnectedLayer(10);
    s1 = softmaxLayer;
    outputLayer = classificationLayer();
    layers = [inputLayer; c1; p1; c2; p2; f1; f2; s1;outputLayer];
    opts = trainingOptions('sgdm');
    t = categorical(t);
    t = trainNetwork(td,t,layers,opts);
end