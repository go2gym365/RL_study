# λ = 0
python cs285/scripts/run_hw2.py \
  --env_name LunarLander-v2 --ep_len 1000 \
  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
  --use_reward_to_go --use_baseline --gae_lambda 0 \
  --exp_name lunar_lander_lambda0

# λ = 0.95
python cs285/scripts/run_hw2.py \
  --env_name LunarLander-v2 --ep_len 1000 \
  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
  --use_reward_to_go --use_baseline --gae_lambda 0.95 \
  --exp_name lunar_lander_lambda095

# λ = 0.98
python cs285/scripts/run_hw2.py \
  --env_name LunarLander-v2 --ep_len 1000 \
  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
  --use_reward_to_go --use_baseline --gae_lambda 0.98 \
  --exp_name lunar_lander_lambda098

# λ = 0.99
python cs285/scripts/run_hw2.py \
  --env_name LunarLander-v2 --ep_len 1000 \
  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
  --use_reward_to_go --use_baseline --gae_lambda 0.99 \
  --exp_name lunar_lander_lambda099

# λ = 1
python cs285/scripts/run_hw2.py \
  --env_name LunarLander-v2 --ep_len 1000 \
  --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
  --use_reward_to_go --use_baseline --gae_lambda 1 \
  --exp_name lunar_lander_lambda1