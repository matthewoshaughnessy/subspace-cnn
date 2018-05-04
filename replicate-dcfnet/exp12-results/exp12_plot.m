% --- parameters ---
ma_filt_length = 1;


% --- load data ---
variants = {'old','new'};
variantNums = 1:5;
for i = 1:length(variants)
  for k = variantNums
    filename = ['out' num2str(k) '_' variants{i} '.mat'];
    load(filename);
    eval(sprintf(['%s(%d) = struct(' ...
      '''loss_history'',loss_history,' ...
      '''testaccuracy_history'',testaccuracy_history,' ...
      '''time_history'',time_history,' ...
      '''coeff'',[coeff_1.'' coeff_2.'' coeff_3.'']);'],variants{i},k));
  end
end


% --- parse data ---
[nTrain,nEpoch] = size(old(1).loss_history);
ma_filt = 1/ma_filt_length*ones(1,ma_filt_length);
for i = 1:length(variants)
  for k = variantNums
    eval(sprintf('%s(%d).all_loss = %s(%d).loss_history(:);', ...
      variants{i}, k, variants{i}, k));
    eval(sprintf('%s(%d).ma_loss = filter(ma_filt,1,%s(%d).all_loss);', ...
      variants{i}, k, variants{i}, k));
  end
end


%% display accuracy plots
% loss
figure(1); clf;
cols_loss = cbrewer('qual','Set1',3);
ss = linspace(1,nEpoch,length(old(1).ma_loss));
for k = variantNums
  plot(ss,old(k).ma_loss,'color',cols_loss(1,:)); hold on; grid on;
  plot(ss,new(k).ma_loss,'color',cols_loss(2,:));
end
xlim([1.5 nEpoch]);
xlabel('Epoch'); ylabel('Objective'); set(gca,'fontsize',24);
legend('old','new','location','northeast');
export_fig -transparent exp12_plot1_loss.pdf
% accuracy
figure(2); clf;
cols_acc = cbrewer('qual','Set1',3);
for k = variantNums
  plot(old(k).testaccuracy_history,'color',cols_acc(1,:)); hold on; grid on;
  plot(new(k).testaccuracy_history,'color',cols_acc(2,:));
end
xlim([1 nEpoch]);
xlabel('Epoch'); ylabel('Test accuracy'); set(gca,'fontsize',24);
legend('old','new','location','southeast');
export_fig -transparent exp12_plot2_acc.pdf

