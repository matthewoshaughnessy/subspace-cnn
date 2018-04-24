python3 dcfnet_adv.py out_adv_noproj_nonoise false false dct
python3 dcfnet_adv.py out_adv_proj_nonoise true false dct
python3 dcfnet_testing_only.py out_adv_test_noproj_001 './model_out_adv_noproj_nonoise' 0.001
python3 dcfnet_testing_only.py out_adv_test_proj_001 './model_out_adv_proj_nonoise' 0.001
python3 dcfnet_testing_only.py out_adv_test_noproj_005 './model_out_adv_noproj_nonoise' 0.005
python3 dcfnet_testing_only.py out_adv_test_proj_005 './model_out_adv_proj_nonoise' 0.005
python3 dcfnet_testing_only.py out_adv_test_noproj_01 './model_out_adv_noproj_nonoise' 0.01
python3 dcfnet_testing_only.py out_adv_test_proj_01 './model_out_adv_proj_nonoise' 0.01
python3 dcfnet_testing_only.py out_adv_test_noproj_05 './model_out_adv_noproj_nonoise' 0.05
python3 dcfnet_testing_only.py out_adv_test_proj_05 './model_out_adv_proj_nonoise' 0.05
