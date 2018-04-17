for lr in 0.05 0.01 0.005 0.001
            do for momentum in 1.1 1 0.9 0.8

                do  for lr_decay in 0.5 0.8
                        do
                           for trials in 1 2 3 4 5
                                do
                #do echo "Out_$lr_$momentum_$lr_decay"

                                    filename="out_"$trials"_"$lr"_"$momentum"_"$lr_decay
                                    CUDA_VISIBLE_DEVICES=3 python3 dcfnet.py $filename false false $lr $momentum $lr_decay
                                done
                        done
                done
        done
