% --- get results from file names ---
fprintf('Parsing file names...');
filenames = dir('out_*.mat');
filenames = {filenames.name};
nfiles = length(filenames);
params = zeros(nfiles,4);
for i = 1:nfiles
  c = textscan(filenames{i},'out_%f_%f_%f_%f.mat');
  params(i,:) = [c{:}];
end
fprintf('done!\n');

% --- load results ---
fprintf('Loading results');
load(filenames{1},'loss_history','testaccuracy_history');
all_loss = zeros(numel(loss_history),nfiles);
all_acc = zeros(numel(testaccuracy_history),nfiles);
for i = 1:nfiles
  fprintf('.');
  load(filenames{i},'loss_history','testaccuracy_history');
  all_loss(:,i) = loss_history(:);
  all_acc(:,i) = testaccuracy_history(:);
end
fprintf('done!\n');

% --- parse data ---
malength = 100;
values.trial = sort(unique(params(:,1)));
ntrials = length(values.trial);
values.lr = sort(unique(params(:,2)));
nlr = length(values.lr);
values.momentum = sort(unique(params(:,3)));
nmomentum = length(values.momentum);
values.lr_decay = sort(unique(params(:,4)));
nlrdecay = length(values.lr_decay);
ncomb = prod([nlr nmomentum nlrdecay]);
acc = zeros(numel(testaccuracy_history),ntrials,nlr,nmomentum,nlrdecay);
loss = zeros(numel(loss_history),ntrials,nlr,nmomentum,nlrdecay);
fprintf('Parsing results...');
for i = 1:nfiles
  i1 = find(params(i,1) == values.trial);
  i2 = find(params(i,2) == values.lr);
  i3 = find(params(i,3) == values.momentum);
  i4 = find(params(i,4) == values.lr_decay);
  acc(:,i1,i2,i3,i4) = all_acc(:,i);
  loss(:,i1,i2,i3,i4) = conv(all_loss(:,i),1/malength*ones(1,malength),'same');
  fprintf('.');
end
fprintf('done!\n');

% --- plot data ---
export = true;
for imom = 1:nmomentum
  for ilrd = 1:nlrdecay
    % --- plot accuracy ---
    cols = cbrewer('seq','OrRd',5);
    figure(1); clf;
    for i = 1:nlr
      hs = plot(acc(:,:,i,imom,ilrd),'color',cols(i+1,:)); hold on; grid on;
      h(i) = hs(1);
    end
    legend(h, ...
      cellfun(@(x)sprintf('lr = %.3f',x),num2cell(values.lr), ...
      'uniformoutput',false), ...
      'location', 'southeast');
    xlabel('Epoch'); ylabel('Test accuracy'); axis([1 size(all_acc,1) 0 0.9]);
    title({'Test Accuracy', sprintf('(momentum = %.2f, lr decay = %.2f)', ...
      values.momentum(imom), values.lr_decay(ilrd)), ...
      sprintf('max accuracy: %.3f',max(max(max(acc(:,:,:,imom,ilrd)))))});
    set(gca,'fontsize',28);
    set(gcf,'color','white');
    drawnow;
    if export
      exportname = sprintf('exp6b_acc_mom%.1f_lrd%.1f', ...
        values.momentum(imom), values.lr_decay(ilrd));
      exportname(exportname == '.') = [];
      fprintf('Exporting %s.pdf...', exportname);
      export_fig([exportname '.pdf']);
      fprintf('done!\n');
    end
    % --- plot loss ---
    cols = cbrewer('seq','YlGnBu',5);
    figure(2); clf;
    for i = 1:nlr
      hs = plot(linspace(0,1,size(loss,1))*size(all_acc,1), ...
        loss(:,:,i,imom,ilrd),'color',cols(i+1,:));
      hold on; grid on;
      h(i) = hs(1);
    end
    legend(h, ...
      cellfun(@(x)sprintf('lr = %.3f',x),num2cell(values.lr), ...
      'uniformoutput',false), ...
      'location', 'southwest');
    xlabel('Epoch'); ylabel('Objective'); axis([0.2 24.8 0 2.5]);
    title({'Objective value', sprintf('(momentum = %.2f, lr decay = %.2f)', ...
      values.momentum(imom), values.lr_decay(ilrd)), ...
      sprintf('min loss: %.3f',min(min(min(loss(:,:,:,imom,ilrd)))))});
    set(gca,'fontsize',28);
    set(gcf,'color','white');
    drawnow;
    if export
      exportname = sprintf('exp6a_loss_mom%.1f_lrd%.1f', ...
        values.momentum(imom), values.lr_decay(ilrd));
      exportname(exportname == '.') = [];
      fprintf('Exporting %s.pdf...', exportname);
      export_fig([exportname '.pdf']);
      fprintf('done!\n');
    end
  end
end

