

z={'Blues','BuGn','BuPu','GnBu','Greens','Greys','Oranges','OrRd','PuBu','PuBuGn','PuRd',...
             'Purples','RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'Spectral'};

         
count = 0;         
for lr = [0.05, 0.01, 0.005, 0.001]
    for momentum = [ 0.9, 0.8]
        
        count = count+1;
        col = cbrewer('seq',z{count},10);
        
        figure
        grid on
        
        in_count = 0;
        for lr_decay = [0.5 0.8]
            for trials = [1 2 3 4 5]
                in_count = in_count+1;
                str = strcat('out_',num2str(trials),'_',num2str(lr),'_',num2str(momentum),'_',num2str(lr_decay),'.mat');
                load(str)
                hold on, plot(testaccuracy_history*100,'color',col(in_count,:),'LineWidth',2)
            end
        end
        ylabel('test accuracy')
        xlabel('Number of epochs')
        title(strcat('Learning rate: ',num2str(lr),'momentum: ',num2str(momentum)))
        legend({'0.5,1','0.5,2','0.5,3','0.5,4','0.5,5','0.8,1','0.8,2','0.8,3','0.8,4','0.8,5'})
        
    end
end