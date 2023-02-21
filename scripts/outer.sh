run() {
sbatch --output=$log_dir/%j-%x.out -N 1 -n 1 --cpus-per-task=10 --mem=100G --gres=gpu:a100:1 --time 0-23:59:59 --job-name CoFi <<EOF
#!/bin/sh
bash $n/space2/CoFiPruning/scripts/run_CoFi_clm.sh $1 $2 $3 $4
EOF
}

extract(){
    jq -r '[."Books3-valid-1.536M"."eval_sparsity", 
        ."Books3-valid-1.536M"."eval_Books3-valid-1.536M_ppl",
        ."Github-valid-1.536M"."eval_Github-valid-1.536M_ppl", 
        ."FreeLaw-valid-1.536M"."eval_FreeLaw-valid-1.536M_ppl",
        ."Wikipedia-valid-1.536M"."eval_Wikipedia-valid-1.536M_ppl"] | @tsv' $1 
}

type=extract
out_dir=/scratch/gpfs/mengzhou/space2/out/test_round3_fp16_mp
log_dir=/scratch/gpfs/mengzhou/space2/logs
mkdir -p $log_dir
for train_num in 10000 100000; do
for sparsity in 0.3 0.6 0.9; do
for size in 1.3b; do
for freeze_type in emb mm; do
if [[ $freeze_type == "emb" ]]; then suffix="false true";
else suffix="true false"; fi
if [[ $type == "run" ]]; then run $train_num $sparsity $size $suffix;
else dir=$out_dir/CoFi_${train_num}_${sparsity}_opt-${size}_f${freeze_type}; extract $dir/eval_results.json; fi
done
done
done
done



# bash $n/space2/CoFiPruning/scripts/run_CoFi_clm.sh ${train_num} ${lag_lr} ${sparsity} ${size}
