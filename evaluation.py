import inspect
import os
import pdb
import random
import sys
import time

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.data.data_collator import (DataCollator,
                                             DataCollatorWithPadding,
                                             default_data_collator)
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import EvalPrediction

from models.modeling_bert import (CoFiBertForQuestionAnswering,
                                  CoFiBertForSequenceClassification)
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from utils.cofi_utils import *
from utils.qa_utils import *
from utils.utils import *


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def _remove_unused_columns(dataset: "datasets.Dataset", description):
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += ["label", "label_ids"]
    columns = [k for k in signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dset_description = "" if description is None else f"in the {description} set "
    print(
        f"The following columns {dset_description} don't have a corresponding argument in `{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
    )
    dataset.set_format(type=dataset.format["type"], columns=columns)


def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=batch_size,
                            collate_fn=default_data_collator)
    return dataloader

def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v}
                             for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex[answer_column_name]}
                  for ex in datasets["validation"]]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def evaluate(model):
    metrics = {}
    total_infer_times = 0

    t = 2 if task_name in ["squad", "qqp"] else 5
    if task_name in ["rte", "stsb", "cola", "mrpc"]:
        t = 20
    assert t > 1

    total_examples = 0
    for i in range(t):
        _remove_unused_columns(dataset, "evaluation")

        preds = None
        label_ids = None
        total_infer_time = 0
        print(f"Round {i}: There are {len(dataloader)} batches in the dataset.")
        for num_batch, inputs in enumerate(dataloader):
            labels = inputs["labels"] if "labels" in inputs else None
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            with torch.no_grad():
                a = time.time()
                if task_name == "squad":
                    output = model(**inputs)
                    logits = output["start_logits"], output["end_logits"]
                else:
                    logits = model(**inputs)["logits"]
                torch.cuda.synchronize()
                b = time.time()
                total_infer_time += (b-a)
                if i == 0:
                    total_examples += len(logits)
                    preds = logits if preds is None else nested_concat(
                        preds, logits)
                    label_ids = labels if label_ids is None else nested_concat(
                        label_ids, labels)
        if label_ids is not None:
            final_label_ids = nested_numpify(label_ids)
        if preds is not None:
            final_preds = nested_numpify(preds)

        if i == 0:
            metrics["num_examples"] = total_examples

        if i > 0:
            total_infer_times += total_infer_time
    if task_name == 'squad':
        dataset.set_format(
            type=dataset.format["type"], columns=list(dataset.features.keys()))
        eval_preds = post_processing_function(
            eval_examples, dataset, final_preds)
        metrics = compute_metrics(eval_preds)
    else:
        metrics = compute_metrics(EvalPrediction(
            predictions=final_preds, label_ids=final_label_ids))
    total_infer_time = round(total_infer_times / (t-1), 4)
    metrics["seconds/example"] = total_infer_times / (t-1) / total_examples
    return metrics


def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    max_length = 384
    doc_stride = 128
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
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
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def glue_preprocess_function(examples):
    # Tokenize the texts
    sentence1_key, sentence2_key = task_to_keys[task_name]
    max_seq_length = 128
    padding = "max_length"
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
    )

    result = tokenizer(*args, padding=padding,
                       max_length=max_seq_length, truncation=True)
    if task_name == "mnli" and model_name_or_path.startswith("princeton-nlp/"):
        # legacy issue of using GLUEDataset
        label_to_id = {1:2, 0:1, 2:0}
        labels = [label_to_id[i] for i in examples["label"]]
        result["label"] = labels
    return result


def warmup():
    time1 = time.time()
    input = torch.randn(128, 1024).cuda()
    linear = torch.nn.Linear(1024, 1024).cuda()
    for i in range(10000):
        input = linear(input)

    time2 = time.time()
    print(round(time2 - time1, 2), "seconds for warmup")

def get_glue_metric():
    metric = load_metric("glue", task_name)
    is_regression = task_name == "stsb"

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    return compute_metrics

if __name__ == '__main__':
    # warmup
    warmup()

    # data
    task_name = sys.argv[1].lower()
    model_name_or_path = sys.argv[2]
    bs = 128

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True if task_name == "squad" else False, padding_side="right", truncation_size="right")

    if task_name != "squad":
        # data_args = DataTrainingArguments(task_name=task_name,
        #   data_dir=os.path.join(data_dir, task_name))
        # dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
        if task_name == "mnli":
            set_name = "validation_matched"
        else:
            set_name = "validation"
        dataset = datasets.load_dataset("glue", task_name)[set_name]
        dataset = dataset.map(glue_preprocess_function, batched=True)

        compute_metrics = get_glue_metric()
    else:
        metric = load_metric("squad")

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        datasets = load_dataset("squad")
        column_names = datasets["validation"].column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"
        dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
        )
        eval_examples = datasets["validation"]

    dataloader = get_dataloader(dataset, bs)

    # load model
    if "squad" in task_name:
        model_class = CoFiBertForQuestionAnswering
    else:
        model_class = CoFiBertForSequenceClassification

    zs = load_zs(model_name_or_path)

    # for compressed models
    if zs is None:
        model = model_class.from_pretrained(model_name_or_path)
    # for full models with compression vectors zs
    else:
        model = load_model(model_name_or_path, model_class, zs)

    model = model.cuda()
    model = model.eval()

    model.config.output_hidden_states = False
    model.config.output_attentions = False

    metrics = evaluate(model)
    model_size = calculate_parameters(model)
    full_model_size = calculate_parameters(model_class(model.config))
    sparsity = 1 - round(model_size / full_model_size, 3)

    print(f"Task: {task_name}")
    print(f"Model path: {model_name_or_path}")
    print(f"Model size: {model_size}")
    print(f"Sparsity: {sparsity}")
    for key in metrics:
        print(f"{key}: {round(metrics[key], 6 if 'seconds' in key else 4)}")
    print()
