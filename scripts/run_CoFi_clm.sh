#!/bin/bash
#SBATCH --job-name=CoFi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH -t 11:00:00

## base functions
function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

declare -A BS=(["opt-125m"]=10 ["opt-350m"]=10 ["opt-1.3b"]=10 ["opt-13b"]=1 ["opt-66b"]=1)
if [[ $(nvidia-smi | grep MiB | head -n 1 | grep 40960) ]]; then gpu=gpu40; else gpu=gpu80; fi
export CUDA_VISIBLE_DEVICES=0
# ********** Scripts start here ***********

## proj dirs
proj_dir=/scratch/gpfs/mengzhou/space2
code_dir=/${proj_dir}/CoFiPruning
output_dir=${proj_dir}/out


## subdir
subdir=test_round3_fp16_mp # test_round3_fp16_mp # test_round2_lr_scheduler
test=false
resume_training=false
max_train_samples=$1
freeze_main_model=$4
freeze_embeddings=$5

# data
train_domains=(Books3)
eval_domains=(Books3 Github FreeLaw Wikipedia)
data_name=$(join_by - ${train_domains[@]})

## base params
model_name_or_path=facebook/opt-${3}
model_name=$(basename $model_name_or_path)
batch_size=${BS[$model_name]}
if [[ $gpu == gpu40 ]]; then batch_size=$(( batch_size / 2 )); fi
grad_accu_steps=$(( 10 / batch_size ))
learning_rate=1e-4
epochs=1

## pruning params
pruning_modules="head head_layer intermediate mlp"
reg_learning_rate=0.1
prepruning_finetune_epochs=0
lagrangian_warmup_epochs=0.2
joint_training_epochs=0.8
final_finetune_epochs=0
target_sparsity=$2

## distillation params
distillation_options=None

##  TODO: change to 50
## base arguments

base_arguments="python3 $code_dir/run_clm.py \
      --logging_steps 50 \
      --save_strategy no \
      --save_steps 500 \
      --eval_steps 50 \
      --evaluation_strategy steps \
      --model_name_or_path ${model_name_or_path} \
      --do_train \
      --do_eval \
      --per_device_train_batch_size ${batch_size} \
      --gradient_accumulation_steps ${grad_accu_steps} \
      --per_device_eval_batch_size 32 \
      --learning_rate ${learning_rate} \
      --num_train_epochs ${epochs} \
      --report_to wandb \
      --max_eval_samples 100 \
      --max_train_samples ${max_train_samples} \
      --warmup_ratio 0.2 \
      --block_size 512 \
      --fp16 "
      # --fsdp auto_wrap " 

if [[ $freeze_embeddings == true ]]; then base_arguments="$base_arguments --freeze_embeddings "; suffix=femb; fi
if [[ $freeze_main_model == true ]]; then base_arguments="$base_arguments --freeze_main_model "; suffix=fmm; fi

## Data
THE_PILE=/scratch/gpfs/mengzhou/space6/data/the_pile
base_arguments="$base_arguments --preprocessed_train_files " # $THE_PILE/category/Books3-train-0.5B.pt $THE_PILE/category/Github-train-0.5B.pt --processed_validation_files $THE_PILE/category/val/FreeLaw-valid-1.536M.pt $THE_PILE/category/val/Wikipedia-valid-1.536M.pt "
for train_domain in ${train_domains[@]}; do base_arguments="$base_arguments $THE_PILE/category/${train_domain}-train-0.5B.pt "; done
base_arguments+="--preprocessed_validation_files "
for eval_domain in ${eval_domains[@]}; do base_arguments="$base_arguments $THE_PILE/category/val/${eval_domain}-valid-1.536M.pt "; done


## pruning
if [[ ${pruning_modules} != None ]]; then
   run_name=CoFi_${max_train_samples}_${target_sparsity}_${model_name}
   run_name=${run_name}_${suffix}
   base_arguments="$base_arguments --pruning_modules ${pruning_modules} \
                                   --reg_learning_rate ${reg_learning_rate} \
                                   --prepruning_finetune_epochs ${prepruning_finetune_epochs} \
                                   --lagrangian_warmup_epochs ${lagrangian_warmup_epochs} \
                                   --joint_training_epochs ${joint_training_epochs} \
                                   --final_finetune_epochs ${final_finetune_epochs} \
                                   --target_sparsity ${target_sparsity} "
fi

## Distillation
if [[ ${distillation_options} != None ]]; then
   run_name=distillation
   run_name=${run_name}_${suffix}
   base_arguments="$base_arguments --distill_layer_loss_alpha ${distill_layer_alpha} \
                                   --distillation_options ${distillation_options} \
                                   --distill_ce_loss_alpha ${distill_ce_alpha} \
                                   --distillation_path ${distillation_path} \ 
                                   --distill_temp ${distill_temp} \
                                   --layer_distill_version ${layer_distill_version} "
fi

## test
if [ ${test} = true ]; then
    base_arguments="$base_arguments --test \
                                    --max_train_samples 100 \
                                    --max_eval_samples 10 \
                                    --eval_steps 8 "
fi

## output_dir
output_dir=${output_dir}/${subdir}/${run_name}
mkdir -p ${output_dir}
if [[ $resume_training = false ]]; then
base_arguments="$base_arguments --overwrite_output_dir "
fi

## wandb configs
if [ ${test} = true ]; then
    export WANDB_PROJECT=pruning_test
else
    export WANDB_PROJECT=pruning
fi
export WANDB_MODE=dryrun
export WANDB_DIR=$output_dir

## run
base_arguments="${base_arguments} --output_dir ${output_dir} --run_name ${run_name} "

${base_arguments} 2>&1 | tee ${output_dir}/log.txt
