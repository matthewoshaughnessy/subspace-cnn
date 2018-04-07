% --- parameters ---
ma_filt_length = 200;
col = cbrewer('qual','Paired',8);


% --- load data ---
variants = {'noproj_nonoise','noproj_noise','proj_nonoise','proj_noise'};
for i = 1:length(variants)
  filename = ['experiment1_out_' variants{i} '.mat'];
  load(filename);
  eval([variants{i} ' = struct(' ...
    '''loss_history'',loss_history,' ...
    '''testaccuracy_history'',testaccuracy_history,' ...
    '''time_history'',time_history,' ...
    '''coeff'',[coeff_1.'' coeff_2.'' coeff_3.'']);']);
  clearvars('-except','ma_filt_length','col','variants',variants{:});
end


%%% --- parse data ---
for f = fieldnames(noproj_nonoise)
  assert(isequal(size(noproj_nonoise.(f{1})), ...
    size(noproj_noise.(f{1})), ...
    size(proj_nonoise.(f{1})), ...
    size(proj_noise.(f{1}))));
end
[nTrain,nEpoch] = size(noproj_nonoise.loss_history);
ma_filt = 1/ma_filt_length*ones(1,ma_filt_length);
for i = 1:length(variants)
  eval([variants{i} '.all_loss = ' variants{i} '.loss_history(:);']);
  eval([variants{i} '.ma_loss = filter(ma_filt,1,' variants{i} '.all_loss);']);
end


%% --- plot loss vs epoch ---
figure(1); clf;
nsubsamp = 100;
ii = (0:nEpoch*nTrain-1)/nTrain;
ss = 1:nsubsamp:nEpoch*nTrain;
plot(ii(ss),noproj_nonoise.ma_loss(ss),'color',col(1,:)); hold on; grid on;
plot(ii(ss),proj_nonoise.ma_loss(ss),'color',col(2,:));
plot(ii(ss),noproj_noise.ma_loss(ss),'color',col(3,:));
plot(ii(ss),proj_noise.ma_loss(ss),'color',col(4,:));
set(gca,'fontsize',28);
title('Training loss','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel(sprintf('Loss (length-%d MA)',ma_filt_length),'interpreter','latex');
hl = legend('No noise','No noise, projection','Noisy','Noisy, projection');
set(hl,'interpreter','latex','location','northeast');
export_fig -transparent experiment1_plot1.pdf


%%
figure(2); clf;
plot(1:nEpoch,100*noproj_nonoise.testaccuracy_history,'o-','color',col(1,:));
hold on; grid on;
plot(1:nEpoch,100*proj_nonoise.testaccuracy_history,'o-','color',col(2,:));
plot(1:nEpoch,100*noproj_noise.testaccuracy_history,'o-','color',col(3,:));
plot(1:nEpoch,100*proj_noise.testaccuracy_history,'o-','color',col(4,:));
set(gca,'fontsize',28);
title('Test accuracy','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel('Accuracy (\%)','interpreter','latex');
hl = legend('No noise','No noise, projection','Noisy','Noisy, projection');
set(hl,'interpreter','latex','location','east');
export_fig -transparent experiment1_plot2.pdf

