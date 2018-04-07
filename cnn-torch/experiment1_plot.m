% --- parameters ---
ma_filt_length = 200;
col = cbrewer('qual','Paired',8);


% --- load data ---
load experiment1_out_proj.mat
proj = struct( ...
  'loss_history', loss_history, ...
  'testaccuracy_history', testaccuracy_history, ...
  'time_history', time_history, ...
  'coeff', [coeff_1.' coeff_2.' coeff_3.']);
clearvars -except proj ma_filt_length col
load experiment1_out_noproj.mat
noproj = struct( ...
  'loss_history', loss_history, ...
  'testaccuracy_history', testaccuracy_history, ...
  'time_history', time_history, ...
  'coeff', [coeff_1.' coeff_2.' coeff_3.']);
clearvars -except proj noproj ma_filt_length col


% --- parse data ---
for f = fieldnames(proj)
  assert(isequal(size(proj.(f{1})),size(noproj.(f{1}))));
end
[nTrain,nEpoch] = size(proj.loss_history);
proj.all_loss = proj.loss_history(:);
noproj.all_loss = noproj.loss_history(:);
ma_filt = 1/ma_filt_length*ones(1,ma_filt_length);
proj.ma_loss = filter(ma_filt,1,proj.all_loss);
noproj.ma_loss = filter(ma_filt,1,noproj.all_loss);


% --- plot loss vs epoch ---
figure(1); clf;
nsubsamp = 100;
ii = (0:nEpoch*nTrain-1)/nTrain;
h1 = plot(ii(1:nsubsamp:end),proj.all_loss(1:nsubsamp:end),'linew',0.5,'color',col(1,:));
hold on; grid on;
h2 = plot(ii(1:nsubsamp:end),noproj.all_loss(1:nsubsamp:end),'linew',0.5,'color',col(3,:));
h3 = plot(ii,proj.ma_loss,'linew',1,'color',col(2,:));
h4 = plot(ii,noproj.ma_loss,'linew',1,'color',col(4,:));
set(gca,'fontsize',16);
title('Training loss','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel(sprintf('Loss (length-%d MA)',ma_filt_length),'interpreter','latex');
hl = legend([h3 h4],'projection','no projection');
set(hl,'interpreter','latex','location','northeast');

%%
figure(2); clf;
plot(1:nEpoch,100*proj.testaccuracy_history,'o-','linew',1.5,'color',col(2,:));
hold on; grid on;
plot(1:nEpoch,100*noproj.testaccuracy_history,'o-','linew',1.5,'color',col(4,:));
set(gca,'fontsize',16);
title('Test accuracy','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel('Accuracy (\%)','interpreter','latex');
hl = legend('projection','no projection');
set(hl,'interpreter','latex','location','southeast');

