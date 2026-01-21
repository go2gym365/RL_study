# No baseline
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --exp_name cheetah
# Baseline
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

# Baseline + bgs 1
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 1 --exp_name cheetah_baseline_bgs1

# Baseline + advantage normalization
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 5 -na --exp_name cheetah_baseline_na