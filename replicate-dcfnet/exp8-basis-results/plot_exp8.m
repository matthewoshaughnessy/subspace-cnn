% --- parameters ---
ma_filt_length = 1;
col = cbrewer('qual','Set1',8);


% --- load data ---
variants = {'rb_noise','dct_noise','rb_nonoise','dct_nonoise'};
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
[nTrain,nEpoch] = size(rb_noise(1).loss_history);
ma_filt = 1/ma_filt_length*ones(1,ma_filt_length);
for i = 1:length(variants)
  for k = variantNums
    eval(sprintf('%s(%d).all_loss = %s(%d).loss_history(:);', ...
      variants{i}, k, variants{i}, k));
    eval(sprintf('%s(%d).ma_loss = filter(ma_filt,1,%s(%d).all_loss);', ...
      variants{i}, k, variants{i}, k));
  end
end


% --- calculate accuracy ---
dct_nonoise_acc = mean([dct_nonoise.testaccuracy_history],2);
dct_noise_acc = mean([dct_noise.testaccuracy_history],2);
rb_nonoise_acc = mean([rb_nonoise.testaccuracy_history],2);
rb_noise_acc = mean([rb_noise.testaccuracy_history],2);


%% display sample convolutional filters
figure(1); clf;
load out1_dct_nonoise
subplot(261); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
subplot(262); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
subplot(263); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_dct_noise
subplot(264); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
subplot(265); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
subplot(266); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_rb_nonoise
subplot(267); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
subplot(268); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
subplot(269); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
load out1_rb_noise
subplot(2,6,10); imagesc(reshape(sum(basis1.*coeff_1,2),[5 5]));
axis square; axis off; colormap gray;
subplot(2,6,11); imagesc(reshape(sum(basis2.*coeff_2,2),[5 5]));
axis square; axis off; colormap gray;
subplot(2,6,12); imagesc(reshape(sum(basis3.*coeff_3,2),[5 5]));
axis square; axis off; colormap gray;
export_fig -transparent exp8_plot1_filters.png


%% display sample bases
figure(2); clf;
load out1_dct_nonoise;
subplot(121); imagesc(basis1); axis square; box on; axis off;
load out1_rb_nonoise;
subplot(122); imagesc(basis1); axis square; box on; axis off;
export_fig -transparent exp8_plot1_bases.png

