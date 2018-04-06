results_train_accuracy = [0.632, 0.658; 0.692, 0.699; 0.715, 0.701; 0.796, 0.773; 0.919, 0.915];
results_test_accuracy  = [0.611, 0.588; 0.587, 0.609; 0.597, 0.584; 0.591, 0.589; 0.565, 0.557];
pct_train = 1:-0.2:0.2;

colors = cbrewer('qual','Set1',4);

clf;
subplot(211);
plot(pct_train,results_train_accuracy(:,1),'o-','linew',2,'color',colors(1,:));
grid on; hold on;
plot(pct_train,results_train_accuracy(:,2),'x-','linew',2,'color',colors(2,:));
legend('Standard','Subspace constrained','location','sw');
%xlabel('Portion of training data unique');
ylabel('Accuracy');
title('Training set');
set(gca,'fontsize',20,'xdir','reverse'); ylim([0.4 1]);

subplot(212);
plot(pct_train,results_test_accuracy(:,1),'o-','linew',2,'color',colors(1,:));
grid on; hold on;
plot(pct_train,results_test_accuracy(:,2),'x-','linew',2,'color',colors(2,:));
legend('Standard','Subspace constrained','location','sw');
xlabel('Portion of training data unique');
ylabel('Accuracy');
title('Test set');
set(gca,'fontsize',20,'xdir','reverse'); ylim([0.5 0.7]);
