import logging
import os
import sys
from copy import deepcopy

import torch
import transformers
from datasets import load_dataset, load_metric
from transformers import (AutoConfig, AutoTokenizer, EvalPrediction,
                          HfArgumentParser, PreTrainedTokenizerFast)
from transformers import SquadDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments, default_data_collator, set_seed

from args import AdditionalArguments
from models.l0_module import L0Module
from models.model_args import ModelArguments
from models.modeling_bert import CoFiBertForQuestionAnswering
from trainer.trainer_qa import CoFiQuestionAnsweringTrainer
from utils.cofi_utils import *
from utils.qa_utils import postprocess_qa_predictions
from utils.utils import *

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, additional_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
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
    logger.info("Training/evaluation.py parameters %s", training_args)

    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    torch.save(additional_args, os.path.join(
        training_args.output_dir, "additional_args.bin"))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # print all arguments
    log_all_parameters(logger, model_args, data_args,
                       training_args, additional_args)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name,
                                data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # set up configuration for distillation
    if additional_args.do_distil:
        config.output_attentions = True
        config.output_hidden_states = True

    Model = CoFiBertForQuestionAnswering

    teacher_model = None
    if additional_args.do_distill:
        teacher_model = Model.from_pretrained(
            additional_args.distillation_path,
            config=deepcopy(config)
        )
        teacher_model.eval()

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # initialize the layer transformation matrix to be an identity matrix
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)

    logger.info(model)
    logger.info(f"Model size: {calculate_parameters(model)}")

    zs = None
    if additional_args.pretrained_pruned_model is not None and not additional_args.pretrained_pruned_model == "None":
        zs = load_zs(additional_args.pretrained_pruned_model)

        model = load_model(additional_args.pretrained_pruned_model, Model, zs)
        prune_model_with_z(zs, model)
        print(
            f"Model Size after pruning and padding: {calculate_parameters(model)}")

    l0_module = None
    if additional_args.str_pruning_method == "l0_reg":
        l0_module = L0Module(config=config,
                             droprate_init=additional_args.droprate_init,
                             temperature=additional_args.temperature,
                             target_sparsity=additional_args.target_sparsity,
                             pruning_type=additional_args.pruning_type)

    # Tokenizer check: this script requires a fast tokenizcser.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.py.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(
                        token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation.py, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        # validation_dataset = load_and_cache_examples(training_args, data_args, model_args, tokenizer, evaluate=True, output_examples=False)
        validation_dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (default_data_collator)

    # Post-processing:
    def post_processing_function(examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]}
                      for ex in datasets["validation"]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric(
        "squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    trainer = CoFiQuestionAnsweringTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        logger=logger,
        l0_module=l0_module,
        teacher_model=teacher_model
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
