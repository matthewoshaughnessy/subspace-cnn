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
  loss(:,i1,i2,i3,i4) = all_loss(:,i);
  fprintf('.');
end
fprintf('done!\n');

% --- plot data ---
for imom = 1:nmomentum
  for ilrd = 1:nlrdecay
    cols = cbrewer('seq','PuRd',5);
    clf;
    for i = 1:nlr
      hs = plot(acc(:,:,i,imom,ilrd),'color',cols(i+1,:)); hold on; grid on;
      h(i) = hs(1);
    end
    legend(h, ...
      cellfun(@(x)sprintf('lr = %.3f',x),num2cell(values.lr), ...
      'uniformoutput',false), ...
      'location', 'southeast');
    xlabel('Epoch'); ylabel('Test accuracy'); ylim([0 0.9]);
    title({'Test Accuracy', sprintf('(momentum = %.2f, lr decay = %.2f)', ...
      values.momentum(imom), values.lr_decay(ilrd)), ...
      sprintf('max accuracy: %.3f',max(max(max(acc(:,:,:,imom,ilrd)))))});
    set(gca,'fontsize',18);
    set(gcf,'color','white');
    drawnow;
    if export
      exportname = sprintf('exp6b_mom%.1f_lrd%.1f', ...
        values.momentum(imom), values.lr_decay(ilrd));
      exportname(exportname == '.') = [];
      fprintf('Exporting %s.pdf...', exportname);
      export_fig([exportname '.pdf']);
      fprintf('done!\n');
    end
  end
end

