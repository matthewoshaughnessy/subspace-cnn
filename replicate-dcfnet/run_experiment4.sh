CUDA_VISIBLE_DEVICES=3 python3 dcfnet.py out_noproj_nonoise false false
CUDA_VISIBLE_DEVICES=3 python3 dcfnet.py out_noproj_noise false true
CUDA_VISIBLE_DEVICES=3 python3 dcfnet.py out_proj_nonoise true false
CUDA_VISIBLE_DEVICES=3 python3 dcfnet.py out_proj_noise true true