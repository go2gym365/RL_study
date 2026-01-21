for seed in $(seq 1 5); do
  python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
    --exp_name pendulum_default_s$seed \
    -rtg --use_baseline -na \
    --batch_size 5000 \
    --seed $seed
done

for seed in $(seq 1 5); do
  python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
    --exp_name pendulum_tuned_bs2000_lr1e2_s$seed \
    -rtg --use_baseline -na \
    --batch_size 2000 -lr 0.01 \
    --seed $seed
done

for seed in $(seq 1 5); do
  python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
    --exp_name pendulum_tuned_bs2000_lr1e2_gae095_s$seed \
    -rtg --use_baseline -na \
    --batch_size 2000 -lr 0.01 \
    --gae_lambda 0.95 \
    --seed $seed
done
