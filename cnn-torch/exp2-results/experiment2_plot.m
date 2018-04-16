% --- parameters ---
ma_filt_length = 5000;
col = cbrewer('qual','Paired',8);


% --- load data ---
% key: [subspace-proj, noisy-test-data, reduced-training, amount-reduced]
variants = {'FFT02','FFT04','FFT06','FFT08','FTT02','FTT04','FTT06', ...
  'FTT08','TFT02','TFT04','TFT06','TFT08','TTT02','TTT04','TTT06','TTT08'};
for i = 1:length(variants)
  filename = ['experiment2_out' variants{i} '.mat'];
  load(filename);
  eval([variants{i} ' = struct(' ...
    '''loss_history'',loss_history,' ...
    '''testaccuracy_history'',testaccuracy_history,' ...
    '''time_history'',time_history,' ...
    '''coeff'',[coeff_1.'' coeff_2.'' coeff_3.'']);']);
  clearvars('-except','ma_filt_length','col','variants',variants{:});
end


%%% --- parse data ---
[nTrain,nEpoch] = size(FFT02.loss_history);
ma_filt = 1/ma_filt_length*ones(1,ma_filt_length);
for i = 1:length(variants)
  eval([variants{i} '.all_loss = ' variants{i} '.loss_history(:);']);
  eval([variants{i} '.ma_loss = filter(ma_filt,1,' variants{i} '.all_loss);']);
end


%% --- 1: training loss, clean data ---
figure(1); clf;
nsubsamp = 1000;
ii = (0:nEpoch*nTrain-1)/nTrain;
ss = 1:nsubsamp:nEpoch*nTrain;
plot(ii(ss),FFT08.ma_loss(ss),'color',col(1,:)); hold on; grid on;
plot(ii(ss),TFT08.ma_loss(ss),'color',col(2,:));
plot(ii(ss),FFT06.ma_loss(ss),'color',col(3,:));
plot(ii(ss),TFT06.ma_loss(ss),'color',col(4,:));
plot(ii(ss),FFT04.ma_loss(ss),'color',col(5,:));
plot(ii(ss),TFT04.ma_loss(ss),'color',col(6,:));
plot(ii(ss),FFT02.ma_loss(ss),'color',col(7,:));
plot(ii(ss),TFT02.ma_loss(ss),'color',col(8,:));
set(gca,'fontsize',20);
title('Training loss, clean data','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel(sprintf('Loss (length-%d MA)',ma_filt_length),'interpreter','latex');
hl = legend('80\%, no proj','80\%, proj', ...
  '60\%, no proj', '60\%, proj', ...
  '40\%, no proj', '40\%, proj', ...
  '20\%, no proj', '20\%, proj');
set(hl,'interpreter','latex','location','northeast');
set(gcf,'color','white'); xlim([1 30]);
export_fig experiment2_plot1.pdf


%% --- 2: training loss, noisy data ---
figure(2); clf;
nsubsamp = 1000;
ii = (0:nEpoch*nTrain-1)/nTrain;
ss = 1:nsubsamp:nEpoch*nTrain;
plot(ii(ss),FTT08.ma_loss(ss),'color',col(1,:)); hold on; grid on;
plot(ii(ss),TTT08.ma_loss(ss),'color',col(2,:));
plot(ii(ss),FTT06.ma_loss(ss),'color',col(3,:));
plot(ii(ss),TTT06.ma_loss(ss),'color',col(4,:));
plot(ii(ss),FTT04.ma_loss(ss),'color',col(5,:));
plot(ii(ss),TTT04.ma_loss(ss),'color',col(6,:));
plot(ii(ss),FTT02.ma_loss(ss),'color',col(7,:));
plot(ii(ss),TTT02.ma_loss(ss),'color',col(8,:));
set(gca,'fontsize',20);
title('Training loss, noisy data','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel(sprintf('Loss (length-%d MA)',ma_filt_length),'interpreter','latex');
hl = legend('80\%, no proj','80\%, proj', ...
  '60\%, no proj', '60\%, proj', ...
  '40\%, no proj', '40\%, proj', ...
  '20\%, no proj', '20\%, proj');
set(hl,'interpreter','latex','location','northeast');
set(gcf,'color','white'); xlim([1 30]);
export_fig experiment2_plot2.pdf


%% --- 3: accuracy, clean data ---
figure(3); clf;
plot(FFT08.testaccuracy_history,'o-','color',col(1,:)); hold on; grid on;
plot(TFT08.testaccuracy_history,'o-','color',col(2,:));
plot(FFT06.testaccuracy_history,'o-','color',col(3,:));
plot(TFT06.testaccuracy_history,'o-','color',col(4,:));
plot(FFT04.testaccuracy_history,'o-','color',col(5,:));
plot(TFT04.testaccuracy_history,'o-','color',col(6,:));
plot(FFT02.testaccuracy_history,'o-','color',col(7,:));
plot(TFT02.testaccuracy_history,'o-','color',col(8,:));
set(gca,'fontsize',20);
title('Test accuracy, clean data','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel('Accuracy','interpreter','latex');
hl = legend('80\%, no proj','80\%, proj', ...
  '60\%, no proj', '60\%, proj', ...
  '40\%, no proj', '40\%, proj', ...
  '20\%, no proj', '20\%, proj');
set(hl,'interpreter','latex','location','eastoutside');
set(gcf,'color','white'); xlim([1 30]);
export_fig experiment2_plot3.pdf


%% --- 4: accuracy, noisy data ---
figure(3); clf;
plot(FTT08.testaccuracy_history,'o-','color',col(1,:)); hold on; grid on;
plot(TTT08.testaccuracy_history,'o-','color',col(2,:));
plot(FTT06.testaccuracy_history,'o-','color',col(3,:));
plot(TTT06.testaccuracy_history,'o-','color',col(4,:));
plot(FTT04.testaccuracy_history,'o-','color',col(5,:));
plot(TTT04.testaccuracy_history,'o-','color',col(6,:));
plot(FTT02.testaccuracy_history,'o-','color',col(7,:));
plot(TTT02.testaccuracy_history,'o-','color',col(8,:));
set(gca,'fontsize',20);
title('Test accuracy, noisy data','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel('Accuracy','interpreter','latex');
hl = legend('80\%, no proj','80\%, proj', ...
  '60\%, no proj', '60\%, proj', ...
  '40\%, no proj', '40\%, proj', ...
  '20\%, no proj', '20\%, proj');
set(hl,'interpreter','latex','location','eastoutside');
set(gcf,'color','white'); xlim([1 30]);
export_fig experiment2_plot4.pdf

