from dataclasses import dataclass, field
from typing import Optional, List


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
class AdditionalArguments:
    none_or_str = lambda x: None if x == "None" else x
    
    test: bool = field(default=False, metadata={"help": "Testing additional arguments."})
    
    pruning_modules: List[str] = field(default_factory=list, metadata={"help": "Components to be added for pruning. [intermediate|mlp|head|head_layer|hidden]"})
    reg_learning_rate: float = field(default=0.1, metadata={"help": "Learning rate for l0 and lagrangian parameters."})
    scheduler_type: str = field(default="linear", metadata={"help": "type of scheduler"})
    l0_optim: str = field(default="adamw_hf", metadata={"help": "Optimizer for l0 and lagrangian parameters."})
    # 
    # which part to train, if both args are false, we train the whole model
    freeze_main_model: bool = field(default=False, metadata={"help": "Whether we should freeze the main model."})
    freeze_embeddings: bool = field(default=False, metadata={"help": "Whether we should freeze the embeddings."})

    # for L0 module
    droprate_init: float = field(default=0.5, metadata={"help": "Init parameter for loga"})
    temperature: float = field(default=2./3., metadata={"help": "Temperature controlling hard concrete distribution"})
    
    # for training pruning process
    prepruning_finetune_epochs: float = field(default=1, metadata={"help": "Finetuning epochs before pruning."})
    lagrangian_warmup_epochs: float = field(default=2, metadata={"help": "Number of epochs for lagrangian warmup."})
    joint_training_epochs: float = field(default=1, metadata={"help": "Number of epochs for joint training after sparsity level is achieved."})
    final_finetune_epochs: float = field(default=1, metadata={"help": "Number of epochs for final finetuning, while fixing the masks."})
    
    # pruning sparsity
    target_sparsity: float = field(default=0, metadata={"help": "Target sparsity (pruned percentage)"})
    sparsity_epsilon: float = field(default=0, metadata={"help": "Epsilon for sparsity"})

    # distillation setup
    distillation_path: str = field(default=None, metadata={"help": "Path of the teacher model for distillation."})
    distill_options: List[str] = field(default_factory=list, metadata={"help": "Distillation options, [ce|layer]"})
    layer_distill_version: int = field(default=1, metadata={"help": "1: add loss to each layer, 2: add loss to existing layers only"})
    distill_ce_alpha: float = field(default=0.9, metadata={"help": "Distillation ce weight"})
    distill_layer_alpha: float = field(default=0.1, metadata={"help": "Distillation layer weight"})
    distill_temp: float = field(default=2./3., metadata={"help": "Distillation temperature"})
    
    # data
    download_only: Optional[bool] = field(default=False, metadata={"help": "Whether to download only. The della cluster only has access to the Internet on the head node."})
    preprocessed_train_files: List[str] = field(default_factory=list)
    preprocessed_validation_files: List[str] = field(default_factory=list)
    training_data_strategy: str = field(default="merge", metadata={"help": "Strategy when having multiple training datasets. e.g., merge, split"})

    # scheduler
    l0_lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "Type of scheduler for l0 learning rate."})
    

if __name__ == "__main__":
    from transformers import HfArgumentParser
    # parser = HfArgumentParser(AdditionalArguments)
    # (additional_args,) = parser.parse_args_into_dataclasses()

    parser = HfArgumentParser((AdditionalArguments))
    (additional_args,) = parser.parse_args_into_dataclasses()
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            # raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt", "pt"], "`train_file` should be a csv, a json or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."