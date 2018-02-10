% --- training data parameters ---
dataFileName = 'sharky.wave.points';

% --- architecture parameters ---
params.nUnitsHidden = [10 10]; %linear:[4] - circle:[10] - wave,spirals:[10 10]
params.nUnitsOut = 2;
params.stepSize = 0.15;
params.nEpoch = 1000;

% --- debug parameters ---
debug.drawEvery = 50;
debug.makeVideo = true;

% --- nonlinearity ---
activation  = @(x) 1 ./ (1 + exp(-x));
dactivation = @(x) x.*(1-x);
% tanh     = @(x) (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
% splus    = @(x) log(1 + exp(x));
% relu     = @(x) zeros(size(x)).*(x<0) + x.*(x>=0);
% dtanh    = @(x) 4*exp(2*x) ./ (exp(2*x) + 1).^2; % TODO
% dsplus   = @(x) 1 ./ (1 + exp(-x)); % TODO
% drelu    = @(x) zeros(size(x)).*(x<0) + ones(size(x)).*(x>=0); % TODO


% --- import data ---
importedData  = importdata(dataFileName, '\t', 6);
data.train_in = importedData.data(:, 1:end-1).';
data.class    = importedData.data(:, end).';
data.ndim     = size(data.train_in, 1);
data.ntrain   = size(data.train_in, 2);

% --- initialize ---
classes = -1*ones(size(data.class));
nUnitsIn = data.ndim;
nLayers = 2 + length(params.nUnitsHidden);
nUnits = [nUnitsIn params.nUnitsHidden params.nUnitsOut];
mse = -1*ones(1,params.nEpoch);
% add biases
nUnits(1:end-1) = nUnits(1:end-1) + 1;
data.train_in = [ones(1,data.ntrain); data.train_in];
% construct desired outputs for training data
data.train_out = zeros(data.ndim, data.ntrain);
for i = 1:data.ntrain
  data.train_out(data.class(i),i) = 1;
end

% --- initialize weights ---
Ws = cell(1, nLayers);
dWs = cell(1, nLayers-1);
for i = 1:nLayers-1
  Ws{i} = 2*rand(nUnits(i+1), nUnits(i)) - 1;
  Ws{i}(1,:) = 0;
  dWs{i} = zeros(nUnits(i+1), nUnits(i));
end
Ws{end} = ones(nUnits(end), 1); % initialize weights for output nodes

% --- initialize activations ---
as = cell(1, nLayers);
for i = 1:length(as)
  as{i} = zeros(nUnits(i), 1);
end
bpErrs = as;

% --- train ---
for iepoch = 1:params.nEpoch
  
  for isample = 1:data.ntrain
    
    % --- forward pass for sample i---
    as{1} = data.train_in(:,isample);
    for ilayer = 2:nLayers
      as{ilayer} = activation(Ws{ilayer-1}*as{ilayer-1});
      % bias units aren't connected to previous layer
      if (ilayer ~= nLayers)
        as{ilayer}(1) = 1;
      end
    end
    
    % --- backward pass for sample i---
    bpErrs{end} = data.train_out(:,isample) - as{end};
    for ilayer = nLayers-1:-1:1
      grad = dactivation(as{ilayer+1});
      bpErrs{ilayer} = Ws{ilayer}'*(bpErrs{ilayer+1}.*grad);
    end
    
    % --- backward pass for sample i, weight change update ---
    for ilayer = nLayers:-1:2
      derivative = dactivation(as{ilayer});
      dWs{ilayer-1} = dWs{ilayer-1} + (bpErrs{ilayer} .* derivative) * as{ilayer-1}.';
    end
    
  end
    
  % --- backward pass weight update ---
  for ilayer = 1:nLayers-1
    Ws{ilayer} = Ws{ilayer} + params.stepSize*dWs{ilayer};
    dWs{ilayer} = 0 * dWs{ilayer};
  end
  
  % --- calculate MSE ---
  for isample = 1:data.ntrain
    out = feedForward(data.train_in(:,isample), Ws, activation);
    [~,classes(isample)] = max(out);
  end
  mse(iepoch) = sum(double(classes~=data.class).^2) / data.ntrain;
  
  % --- update plot ---
  fprintf('Epoch %d of %d: MSE = %.4f.\n', iepoch, params.nEpoch, mse(iepoch));
  if mod(iepoch,debug.drawEvery) == 0
    visualizeNetwork(data,Ws,mse(1:iepoch),activation);
  end
  
end
