function visualizeNetwork(data,Ws,mse,activation)

nLayers = length(Ws);
nUnits = zeros(1,nLayers);
for i = 1:nUnits
  nUnits(i) = size(Ws,2) - 1;
end

% plot decision boundary
unique_classes = unique(data.class);
training_colors = {'r.', 'c.'};
subplot(2,2,1);
cla;
hold on;
title('Decision boundary');

margin = 0.05; step = 0.01;
xx = min(data.train_in(2,:))-margin : step : max(data.train_in(2,:))+margin;
yy = min(data.train_in(3,:))-margin : step : max(data.train_in(3,:))+margin;
DecisionBound = zeros(length(yy),length(xx));

for ix = 1:length(xx)
  for iy = 1:length(yy)
    x = xx(ix);
    y = yy(iy);
    out = feedForward([1 x y].', Ws, activation);
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
plot(mse);
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

end