from dataclasses import dataclass, field
from typing import Optional


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

@dataclass
class AdditionalArguments():
    test: bool = field(
        default=False,
        metadata={
            "help": "Testing additional arguments."
        },
    )

    ex_name: str = field(default="test", metadata={"help": "Name of experiment. Base directory of output dir."})
    pruning_type: str = field(default=None, metadata={"help": "Type of pruning"})
    reg_learning_rate: float = field(default=0.1, metadata={"help": "Learning rate for regularization."})
    scheduler_type: str = field(default="linear", metadata={"help": "type of scheduler"})
    freeze_embeddings: bool = field(default=False, metadata={"help": "Whether we should freeze the embeddings."})

    pretrained_pruned_model: str = field(default=None, metadata={"help": "Path of pretrained model."})

    droprate_init: float = field(default=0.5, metadata={"help": "Init parameter for loga"})
    temperature: float = field(default=2./3., metadata={"help": "Temperature controlling hard concrete distribution"})
    prepruning_finetune_epochs: int = field(default=1, metadata={"help": "Finetuning epochs before pruning"})
    lagrangian_warmup_epochs: int = field(default=2, metadata={"help": "Number of epochs for lagrangian warmup"})
    target_sparsity: float = field(default=0, metadata={"help": "Target sparsity (pruned percentage)"})
    sparsity_epsilon: float = field(default=0, metadata={"help": "Epsilon for sparsity"})

    # distillation setup
    distillation_path: str = field(default=None, metadata={"help": "Path of the teacher model for distillation."})
    do_distill: bool = field(default=False, metadata={"help": "Whether to do distillation or not, prediction layer."})
    do_layer_distill: bool = field(default=False, metadata={"help": "Align layer output through distillation"})
    layer_distill_version: int = field(default=1, metadata={"help": "1: add loss to each layer, 2: add loss to existing layers only"})
    distill_loss_alpha: float = field(default=0.9, metadata={"help": "Distillation loss weight"})
    distill_ce_loss_alpha: float = field(default=0.1, metadata={"help": "Distillation cross entrypy loss weight"})
    distill_temp: float = field(default=2./3., metadata={"help": "Distillation temperature"})

    def __post_init__(self):
        if self.pretrained_pruned_model == "None":
            self.pretrained_pruned_model = None
        if self.pruning_type == "None":
            self.pruning_type = None

        

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    
    t_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the training and validation files."}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            self.t_name = self.t_name.lower()
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."