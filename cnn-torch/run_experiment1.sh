python3 experiment1.py false false
mv experiment1_out.txt experiment1_noproj_nonoise.mat
mv experiment1_out.txt experiment1_noproj_nonoise.mat
python3 experiment1.py true false
mv experiment1_out.txt experiment1_proj_nonoise.mat
mv experiment1_out.txt experiment1_proj_nonoise.mat
python3 experiment1.py false true
mv experiment1_out.txt experiment1_noproj_noise.mat
mv experiment1_out.txt experiment1_noproj_noise.mat
python3 experiment1.py true true
mv experiment1_out.txt experiment1_proj_noise.mat
mv experiment1_out.txt experiment1_proj_noise.mat