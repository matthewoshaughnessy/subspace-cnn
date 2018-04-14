load exp3_noproj.mat
layer = 1;

if layer == 1
  W = conv1_weights;
elseif layer == 2
  W = conv2_weights;
end

clf;
sz = size(W);
for i = 1:prod(sz(1:2))
  subplot(sz(1),sz(2),i);
  [i1,i2] = ind2sub(sz(1:2),i);
  imagesc(squeeze(W(i1,i2,:,:)));
  set(gca,'xtick',[],'ytick',[]);
  colormap gray; axis equal; axis off;
end

export_fig experiment3_plot1_noproj.png

%%
load exp3_proj.mat
layer = 1;

if layer == 1
  W = conv1_weights;
elseif layer == 2
  W = conv2_weights;
end

clf;
sz = size(W);
for i = 1:prod(sz(1:2))
  subplot(sz(1),sz(2),i);
  [i1,i2] = ind2sub(sz(1:2),i);
  imagesc(squeeze(W(i1,i2,:,:)));
  set(gca,'xtick',[],'ytick',[]);
  colormap gray; axis equal; axis off;
end

export_fig experiment3_plot1_proj.png

%%
load exp3_noproj.mat
c1_np = coeff_1;
c2_np = coeff_2;
c3_np = coeff_3;
load exp3_proj.mat
c1_p = coeff_1;
c2_p = coeff_2;
c3_p = coeff_3;

clf;
cols = cbrewer('qual','Set1',5);
subplot(231); stem(c1_np,'color',cols(2,:));
subplot(232); stem(c2_np,'color',cols(2,:));
title('Weights projected onto 2D DCT basis, standard CNN','interpreter','latex');
subplot(233); stem(c3_np,'color',cols(2,:));
subplot(234); stem(c1_p,'color',cols(1,:)); 
subplot(235); stem(c2_p,'color',cols(1,:));
title('Weights projected onto 2D DCT basis, subspace-constrained CNN','interpreter','latex');
subplot(236); stem(c3_p,'color',cols(1,:));
for i = 1:6
  subplot(2,3,i);
  axis([1 13*13 -1 1]);
  set(gca,'xtick',1:39:13*13,'ytick',-1:1/2:1,'xticklabel',[],'yticklabel',[],'fontsize',18);
  grid on;
end

export_fig experiment3_plot2.pdf




