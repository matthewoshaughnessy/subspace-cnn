% --- training data parameters ---
dataFileName = 'sharky.circle.points';

% --- architecture parameters ---
params.nUnitsHidden = [10 10]; %linear:[4] - circle:[10] - wave,spirals:[10 10]
params.nUnitsOut = 2;
params.learningRate = 0.15;
params.nEpochMax = 1000;

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
mse = -1*ones(1,params.nEpochMax);
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
dWs = cell(1, nLayers);
for i = 1:length(Ws)-1
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
backpropErrs = as;

%% train
for iepoch = 1:params.nEpochMax
  
  for isample = 1:data.ntrain
    
    % --- forward pass ---
    as{1} = data.train_in(:,isample);
    for ilayer = 2:nLayers
      as{ilayer} = activation(Ws{ilayer-1}*as{ilayer-1});
      if (ilayer ~= nLayers) % bias nodes aren't connected to previous layer
        as{ilayer}(1) = 1;
      end
    end
    
    % --- backward pass ---
    % (as gradient of the bias nodes are zeros, they won't contribute to previous layer errors nor delta_weights)
    backpropErrs{end} = data.train_out(:,isample) - as{end};
    for ilayer = nLayers-1:-1:1
      grad = dactivation(as{ilayer+1});
      for iunit = 1:length(backpropErrs{ilayer})
        backpropErrs{ilayer}(iunit) = sum(backpropErrs{ilayer+1} .* grad .* Ws{ilayer}(:,iunit) );
      end
    end
    
    % --- backward pass delta weights calculation ---
    for ilayer = nLayers:-1:2
      derivative = dactivation(as{ilayer});
      dWs{ilayer-1} = dWs{ilayer-1} + (backpropErrs{ilayer} .* derivative) * as{ilayer-1}.';
    end
    
  end % isample = 1:data.ntrain
  
  
  for ilayer = 1:nLayers
    dWs{ilayer} = params.learningRate * dWs{ilayer};
  end
  
  % --- backward pass weight update ---
  for ilayer = 1:nLayers-1
    Ws{ilayer} = Ws{ilayer} + dWs{ilayer};
    dWs{ilayer} = 0 * dWs{ilayer};
  end
  
  % --- calculate MSE ---
  for isample = 1:data.ntrain
    out = feedForward(data.train_in(:,isample), Ws, activation);
    [~,classes(isample)] = max(out);
  end
  mse(iepoch) = sum(double(classes~=data.class).^2) / data.ntrain;
  
  % --- visualize ---
  fprintf('Epoch %d of %d: MSE = %.4f, learning rate = %.4f.\n', ...
    iepoch, params.nEpochMax, mse(iepoch), params.learningRate);
  
  if mod(iepoch,debug.drawEvery) == 0
    
    fprintf('Visualizing: calculating decision boundary...');
    
    % plot decision boundary
    unique_classes = unique(data.class);
    training_colors = {'r.', 'c.'};
    subplot(2,2,1);
    cla;
    hold on;
    title({sprintf('Epoch %d/%d',iepoch,params.nEpochMax),'Decision boundary'});
    
    margin = 0.05; step = 0.01;
    xx = min(data.train_in(2,:))-margin : step : max(data.train_in(2,:))+margin;
    yy = min(data.train_in(3,:))-margin : step : max(data.train_in(3,:))+margin;
    DecisionBound = zeros(length(yy),length(xx));
    
    for ix = 1:length(xx)
      for iy = 1:length(yy)
        x = xx(ix);
        y = yy(iy);
        out = feedForward([1 x y].', Ws, activation);
        bound = 1/2;
        DecisionBound(iy,ix) = out(1) > out(2);
      end
    end
    imagesc(xx,yy,DecisionBound); hold on;
    xlim([min(xx) max(xx)]);
    ylim([min(yy) max(yy)]);
    for i = 1:length(unique_classes)
      plot(data.train_in(2,data.class==i), ...
        data.train_in(3,data.class==i), ...
        training_colors{i}, 'markersize', 10);
    end
    axis image; axis off; set(gca,'fontsize',16);
    
    % plot MSE
    fprintf('MSE... ');
    subplot(2,2,3);
    mse(mse==-1) = [];
    plot(mse(1:iepoch));
    ylim([-0.1 0.6]);
    title('Mean square error');
    xlabel('Epoch');
    ylabel('MSE');
    grid on;
    set(gca,'fontsize',16);
    
    % plot weights
    fprintf('weights...');
    for i = 1:nLayers-1
      subplot(nLayers-1,2,2*i);
      imagesc(Ws{i}); colorbar; axis image;
      title(sprintf('W_{%d}',i));
      xlabel(sprintf('layer %d (%d units)',i,nUnits(i)));
      ylabel(sprintf('layer %d (%d units)',i+1,nUnits(i+1)));
      set(gca, ...
        'fontsize', 16, ...
        'xtick', .5:nUnits(i+1), ...
        'ytick', .5:nUnits(i), ...
        'xticklabel', [], ...
        'yticklabel', []);
      grid on;
    end
    
    drawnow;
    fprintf('done!\n');
    
  end
  
end
