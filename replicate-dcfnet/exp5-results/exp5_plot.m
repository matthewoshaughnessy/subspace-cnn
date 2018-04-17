% --- parameters ---
ma_filt_length = 1;
col = cbrewer('qual','Set1',8);


% --- load data ---
variants = cellfun(@(x) ['t' num2str(x)],num2cell(1:5),'uniformoutput',false);
for i = 1:length(variants)
  filename = ['ou' variants{i} '.mat'];
  load(filename);
  eval([variants{i} ' = struct(' ...
    '''loss_history'',loss_history,' ...
    '''testaccuracy_history'',testaccuracy_history,' ...
    '''time_history'',time_history,' ...
    '''coeff'',[coeff_1.'' coeff_2.'' coeff_3.'']);']);
  clearvars('-except','ma_filt_length','col','variants',variants{:});
end


%%% --- parse data ---
for f = fieldnames(t1)
  assert(isequal(size(t1.(f{1})), ...
    size(t2.(f{1})), ...
    size(t3.(f{1})), ...
    size(t4.(f{1}))));
end
[nTrain,nEpoch] = size(t1.loss_history);
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
plot(ii(ss),t1.ma_loss(ss),'color',col(1,:)); hold on; grid on;
plot(ii(ss),t2.ma_loss(ss),'color',col(2,:));
plot(ii(ss),t3.ma_loss(ss),'color',col(3,:));
plot(ii(ss),t4.ma_loss(ss),'color',col(4,:));
plot(ii(ss),t5.ma_loss(ss),'color',col(5,:));
set(gca,'fontsize',28);
title('Training loss','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel(sprintf('Loss (length-%d MA)',ma_filt_length),'interpreter','latex');
hl = legend(cellfun(@(x)['Trial ' num2str(x)],num2cell(1:5),'uniformoutput',false));
set(hl,'interpreter','latex','location','northeast');
set(gca,'yscale','log');
export_fig -transparent exp5_plot1.pdf


%%
figure(2); clf;
plot(1:nEpoch,100*t1.testaccuracy_history,'.-','color',col(1,:));
hold on; grid on;
plot(1:nEpoch,100*t2.testaccuracy_history,'.-','color',col(2,:));
plot(1:nEpoch,100*t3.testaccuracy_history,'.-','color',col(3,:));
plot(1:nEpoch,100*t4.testaccuracy_history,'.-','color',col(4,:));
plot(1:nEpoch,100*t5.testaccuracy_history,'.-','color',col(5,:));
set(gca,'fontsize',28);
title('Test accuracy','interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel('Accuracy (\%)','interpreter','latex');
hl = legend(cellfun(@(x)['Trial ' num2str(x)],num2cell(1:5),'uniformoutput',false));
set(hl,'interpreter','latex','location','southeast');
export_fig -transparent exp5_plot2.pdf

