import math
import os
import sys
import torch
import datasets
import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

from transformers.trainer import Trainer
import logging

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from args import AdditionalArguments, DataTrainingArguments 
from models.model_args import ModelArguments
from trainer.trainer_modified import CoFiTrainer
import wandb
import numpy as np
import logging 

from models.l0_module import L0Module
from models.modeling_opt import CoFiOPTForCausalLM
from cus_data.load_data_hf import load_raw_dataset, preprocess_datasets, load_preprocessed_datasets
from cus_data.dataloader import group_texts
from cus_data.data_mapping import get_metric_mapping

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)

def log_dataset(dataset, description):
    print("*" * 20 + f" {description} datasets " + "*" * 20)
    if isinstance(dataset, dict):
        for key in dataset:
            print(f"{key}: {len(dataset[key])} examples")
    else:
        print(f"{description}: {len(dataset)} examples")
    print()
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Additional arguments {additional_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = CoFiOPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    teacher_model = None
    if additional_args.distillation_path:
        teacher_config = AutoConfig.from_pretrained(additional_args.distillation_path, **config_kwargs)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            additional_args.distillation_path,
            from_tf=bool(".ckpt" in additional_args.distillation_path),
            config=teacher_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        teacher_model.resize_token_embeddings(len(tokenizer))

        # TODO: layer distillation
        model.layer_transformation = torch.nn.Linear(config.hidden_size, teacher_config.hidden_size)
        model.layer_transformation.weight.data.normal_(mean=0.0, std=0.02)
        model.layer_transformation.bias.data.zero_()
        print("Loaded teacher model", additional_args.distillation_path)

    l0_module = None    
    if additional_args.pruning_modules:
        l0_module = L0Module(config,
                             droprate_init=additional_args.droprate_init,
                             target_sparsity=additional_args.target_sparsity,
                             pruning_modules=additional_args.pruning_modules)
        model.l0_module = l0_module

    # load_datasets
    if len(additional_args.preprocessed_train_files) > 0:
        lm_datasets = load_preprocessed_datasets(additional_args)
    else:
        raw_datasets = load_raw_dataset(data_args, model_args, additional_args) 
        lm_datasets = preprocess_datasets(raw_datasets, tokenizer, data_args, training_args, additional_args)
    
    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        # hack to regroup the preprocessed datasets to satisfy the block_size constraint
        if data_args.block_size > len(lm_datasets["train"][0]["input_ids"]):
            train_dataset = train_dataset.map(lambda x: group_texts(x, data_args.block_size), batched=True, batch_size=1000, num_proc=4)
        log_dataset(train_dataset, "train")

    if training_args.do_eval:
        # max eval sample deleted
        eval_dataset = {}
        
        for key in lm_datasets.keys():
            if "valid" in key:
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(data_args.max_eval_samples, len(lm_datasets[key]))
                    eval_dataset[key] = lm_datasets[key].select(range(max_eval_samples))
                else:
                    eval_dataset[key] = lm_datasets[key]
                
                if data_args.block_size > len(eval_dataset[key][0]["input_ids"]):
                    eval_dataset[key] = eval_dataset[key].map(lambda x: group_texts(x, data_args.block_size), batched=True, batch_size=1000, num_proc=4)

        log_dataset(eval_dataset, "eval")
        
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric_mapping = get_metric_mapping()
        if "accuracy" in metric_mapping:
            from metrics.accuracy import Accuracy
            metric = Accuracy()
        else:
            metric = evaluate.load("accuracy")
        def compute_metrics(eval_preds):
            preds = eval_preds[0]
            labels = eval_preds[1]
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    
    if additional_args.download_only:
        sys.exit(0)

    # Initialize our Trainer
    trainer = CoFiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        # l0_module=l0_module,
        teacher_model=teacher_model,
        additional_args=additional_args
    )

    # trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=default_data_collator, compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics)

    # Training
    if training_args.do_train:
        if isinstance(eval_dataset, dict):
            for eval_dataset_name, d in eval_dataset.items():
                metrics = trainer.evaluate(
                    eval_dataset=d,
                    metric_key_prefix=f"eval_{eval_dataset_name}")
        else:
            metrics = trainer.evaluate()
            
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if isinstance(eval_dataset, dict):
            final_metrics = {}
            for eval_dataset_name, d in eval_dataset.items():
                metrics = trainer.evaluate(
                    eval_dataset=d,
                    metric_key_prefix=f"eval_{eval_dataset_name}")
                final_metrics[eval_dataset_name] = metrics
        else:
            final_metrics = trainer.evaluate()

        for key in eval_dataset:
            final_metrics[key + "_eval_samples"] = len(eval_dataset[key])

        # trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", final_metrics)


    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
        # trainer.push_to_hub(**kwargs)
    # else:
        # trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()