% --- parameters ---
ma_filt_length = 1;
col = cbrewer('qual','Paired',8);
ntrials = 5;


% --- load data ---
variants = {'noproj_nonoise','noproj_noise','proj_nonoise','proj_noise'};
for i = 1:length(variants)
  for k = 1:ntrials
    filename = ['out' num2str(k) '_' variants{i} '.mat'];
    load(filename);
    eval([variants{i} '(' num2str(k) ') = struct(' ...
      '''loss_history'',loss_history,' ...
      '''testaccuracy_history'',testaccuracy_history,' ...
      '''time_history'',time_history,' ...
      '''coeff'',[coeff_1.'' coeff_2.'' coeff_3.'']);']);
    clearvars('-except','ma_filt_length','col','variants','i','ntrials',variants{:});
  end
end


%%% --- parse data ---
[nTrain,nEpoch] = size(noproj_nonoise(1).loss_history);
ma_filt = 1/ma_filt_length*ones(1,ma_filt_length);
for i = 1:length(variants)
  for k = 1:ntrials
    eval([variants{i} '(' num2str(k) ')' '.all_loss = ' variants{i} '(' num2str(k) ')' '.loss_history(:);']);
    eval([variants{i} '(' num2str(k) ')' '.ma_loss = filter(ma_filt,1,' variants{i} '(' num2str(k) ')' '.all_loss);']);
  end
end


%% --- plot loss vs epoch ---
figure(1); clf;
nsubsamp = 100;
ii = (0:nEpoch*nTrain-1)/nTrain;
ss = 1:nsubsamp:nEpoch*nTrain;
for i = 1:ntrials
  h(1,i) = plot(ii(ss),noproj_nonoise(i).ma_loss(ss),'color',col(1,:)); hold on; grid on;
  h(2,i) = plot(ii(ss),proj_nonoise(i).ma_loss(ss),'color',col(2,:));
  h(3,i) = plot(ii(ss),noproj_noise(i).ma_loss(ss),'color',col(3,:));
  h(4,i) = plot(ii(ss),proj_noise(i).ma_loss(ss),'color',col(4,:));
end
set(gca,'fontsize',28);
title('Training loss','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel(sprintf('Loss (length-%d MA)',ma_filt_length),'interpreter','latex');
hl = legend([h(1,1) h(2,1) h(3,1) h(4,1)], ...
  'No noise, no projection', ...
  'No noise, projection', ...
  'Noisy, no projection', ...
  'Noisy, projection');
set(hl,'interpreter','latex','location','northeast');
set(gca,'yscale','log');
%export_fig -transparent exp7_plot1.pdf


%%
figure(2); clf;
for i = 1:ntrials
  plot(1:nEpoch,100*noproj_nonoise(i).testaccuracy_history,'^-','color',col(1,:));
  hold on; grid on;
  plot(1:nEpoch,100*proj_nonoise(i).testaccuracy_history,'^-','color',col(2,:));
  plot(1:nEpoch,100*noproj_noise(i).testaccuracy_history,'^-','color',col(3,:));
  plot(1:nEpoch,100*proj_noise(i).testaccuracy_history,'^-','color',col(4,:));
end
set(gca,'fontsize',28);
title('Test accuracy','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel('Accuracy (\%)','interpreter','latex');
hl = legend('No noise, no projection', ...
  'No noise, projection', ...
  'Noisy, no projection', ...
  'Noisy, projection');
set(hl,'interpreter','latex','location','southeast');
%export_fig -transparent exp7_plot2.pdf

