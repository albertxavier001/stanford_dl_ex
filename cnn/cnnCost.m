function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

decayWeight = 0.0;

if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
activationsPooled = cnnPool(poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
% each pixel is a 
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
% 感觉一层隐藏层像是做了一次空间转换
Zd = (Wd * activationsPooled + repmat(bd, 1, size(probs, 2)));
%Zd = bsxfun(@minus, Zd, max(Zd, [], 1));
expZd = exp(Zd);
%expZd(expZd==inf) = 1e200;
%expZd(expZd==0) = 1e-6;
probs = bsxfun(@rdivide, expZd, sum(expZd, 1));
[~,preds] = max(probs,[],1);
preds = preds';

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
%
indImg = 1 : numImages;
indClass = labels';
ind_ytarget = sub2ind(size(probs), indClass, indImg);
ind_ytarget = ind_ytarget';

logProbs = log(probs);
sumCost = logProbs(ind_ytarget);
cost = -sum(sumCost(:)) / numImages;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
%
% soft max
y_target = zeros(size(probs));
y_target(ind_ytarget) = 1;
%
ff2 = probs .^ 2;
ff1 = probs .* y_target;
ff = ff1 - ff2;
%deltaSoft = - (y_target - probs) .* probs .* (1 - probs);
deltaSoft = - (y_target - probs);
%deltaSoft = 0.5 * (y_target - probs).^2;
%deltaSoft = - (y_target - probs) / numImages;
% pool
deltaPool = Wd' * deltaSoft;
sizeDeltaPool = [convDim / poolDim, convDim / poolDim, ...
        numFilters, numImages];
deltaPool  = reshape(deltaPool, sizeDeltaPool);
% conv
deltaUnPool = zeros(convDim, convDim, numFilters, numImages);
for i = 1 : numFilters
    for j = 1 : numImages
        deltaUnPool(:, :, i, j) = 1./ (poolDim.^2) ...
            .* (kron(deltaPool(:, :, i, j), ones(poolDim)));
    end
end

deltaConv = deltaUnPool .* activations .* (1 - activations);
%


%%%

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%activations
Wd_grad = 1./numImages * deltaSoft * activationsPooled' + decayWeight * Wd;
bd_grad = 1./numImages * sum(deltaSoft, 2);
%
for k = 1 : numFilters
    for i = 1 : numImages
        Wc_grad(:, :, k) = Wc_grad(:, :, k) ...
            + conv2(images(:, :, i), ...
            rot90(deltaConv(:, :, k, i), 2), 'valid');
    end
    delConvFilter_k = deltaConv(:, :, k, :);
    bc_grad(k) = sum(delConvFilter_k(:));
end
Wc_grad = 1./numImages * Wc_grad;
bc_grad = 1./numImages * bc_grad;


%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
