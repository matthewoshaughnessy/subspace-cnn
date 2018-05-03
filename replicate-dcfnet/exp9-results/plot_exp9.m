% --- parameters ---
ma_filt_length = 100;


% --- load data ---
variants = {'dim3','dim5','dim7','dim9','dim12','dim15','dim18','dim21'};
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
[nTrain,nEpoch] = size(dim3(1).loss_history);
ma_filt = 1/ma_filt_length*ones(1,ma_filt_length);
for i = 1:length(variants)
  for k = variantNums
    eval(sprintf('%s(%d).all_loss = %s(%d).loss_history(:);', ...
      variants{i}, k, variants{i}, k));
    eval(sprintf('%s(%d).ma_loss = filter(ma_filt,1,%s(%d).all_loss);', ...
      variants{i}, k, variants{i}, k));
  end
end


%% display sample bases
figure(1); clf;
ha = tight_subplot(3,6,[.01 .01],[.01 .01],[.01 .01]);
load out1_dim3
axes(ha(1)); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(7)); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(13)); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_dim5
axes(ha(2)); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(8)); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(14)); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_dim7
axes(ha(3)); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(9)); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(15)); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_dim9
axes(ha(4)); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(10)); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(16)); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_dim12
axes(ha(5)); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(11)); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(17)); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_dim15
axes(ha(6)); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(12)); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
axes(ha(18)); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
set(ha,'XTickLabel',''); set(ha,'YTickLabel','')
export_fig -transparent exp9_plot1_filters.png


%% display accuracy plots
% loss
figure(1); clf;
cols_loss = cbrewer('seq','GnBu',9);
ss = linspace(1,nEpoch,length(dim3(1).ma_loss));
for k = variantNums
  plot(ss,dim3(k).ma_loss,'color',cols_loss(2,:)); hold on; grid on;
  plot(ss,dim5(k).ma_loss,'color',cols_loss(3,:));
  plot(ss,dim7(k).ma_loss,'color',cols_loss(4,:));
  plot(ss,dim9(k).ma_loss,'color',cols_loss(5,:));
  plot(ss,dim12(k).ma_loss,'color',cols_loss(6,:));
  plot(ss,dim15(k).ma_loss,'color',cols_loss(7,:));
  plot(ss,dim18(k).ma_loss,'color',cols_loss(8,:));
  plot(ss,dim21(k).ma_loss,'color',cols_loss(9,:));
end
xlim([1.5 nEpoch]);
xlabel('Epoch'); ylabel('Objective'); set(gca,'fontsize',24);
legend('K = 3','K = 5','K = 7','K = 9','K = 12','K = 15','K = 18','K = 21','location','northeast');
export_fig -transparent exp9_plot2a_loss.pdf
% accuracy
figure(2); clf;
cols_acc = cbrewer('seq','YlOrBr',9);
ss = linspace(1,nEpoch,length(dim3(1).ma_loss));
for k = variantNums
  plot(dim3(k).testaccuracy_history,'color',cols_acc(2,:)); hold on; grid on;
  plot(dim5(k).testaccuracy_history,'color',cols_acc(3,:));
  plot(dim7(k).testaccuracy_history,'color',cols_acc(4,:));
  plot(dim9(k).testaccuracy_history,'color',cols_acc(5,:));
  plot(dim12(k).testaccuracy_history,'color',cols_acc(6,:));
  plot(dim15(k).testaccuracy_history,'color',cols_acc(7,:));
  plot(dim18(k).testaccuracy_history,'color',cols_acc(8,:));
  plot(dim21(k).testaccuracy_history,'color',cols_acc(9,:));
end
xlim([1 nEpoch]);
xlabel('Epoch'); ylabel('Test accuracy'); set(gca,'fontsize',24);
legend('K = 3','K = 5','K = 7','K = 9','K = 12','K = 15','K = 18','K = 21','location','southeast');
export_fig -transparent exp9_plot2b_acc.pdf