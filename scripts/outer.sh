log_dir=/scratch/gpfs/mengzhou/space2/logs
mkdir -p $log_dir
for train_num in 1000 5000 10000; do
for lag_lr in 0.01; do
for sparsity in 0.3 0.6 0.9; do
for size in 1.3b; do
sbatch --output=$log_dir/%j-%x.out -N 1 -n 1 --cpus-per-task=10 --mem=100G --gres=gpu:a100:1 --mail-type=FAIL,TIME_LIMIT --mail-user=mengzhou@cs.princeton.edu --time 0-23:59:59 --job-name CoFi <<EOF
#!/bin/sh
bash $n/space2/CoFiPruning/scripts/run_CoFi_clm.sh ${train_num} ${lag_lr} ${sparsity} ${size}
EOF
done
done
done
done



# bash $n/space2/CoFiPruning/scripts/run_CoFi_clm.sh ${train_num} ${lag_lr} ${sparsity} ${size}
