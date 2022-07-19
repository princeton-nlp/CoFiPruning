import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, EvalPrediction,
                                        EvaluationStrategy, PredictionOutput,
                                        TrainOutput)
from transformers.utils import logging
from transformers.training_args import TrainingArguments

from args import AdditionalArguments
from utils.cofi_utils import *
from utils.utils import *

import wandb

logger = logging.get_logger(__name__)

glue_tasks = {"cola": "matthews_correlation",
              "mnli": "mnli/acc",
              "mrpc": "accuracy",
              "sst2": "accuracy",
              "stsb": "corr",
              "qqp": "accuracy",
              "qnli": "accuracy",
              "rte": "accuracy",
              "sst2_aug": "accuracy",
              "rte_aug": "accuracy",
              "mrpc_aug": "accuracy",
              "qnli_aug": "accuracy",
              "stsb_aug": "corr",}

class Eval_Counter():
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_eval_score = 0
        self.near_sparsity_eval_times = 0
        self.level_best_score = {0.85: 0, 0.8: 0, 0.7: 0,
                                 0.6: 0, 0.75: 0, 0.9: 0, 0.95: 0, 0.65: 0}

    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def update(self, epoch, global_step, eval_score):
        best_so_far = False
        if eval_score > self.best_eval_score:
            self.epoch = epoch
            self.global_step = global_step
            self.best_eval_score = eval_score
            best_so_far = True
        return best_so_far

    def clear(self):
        self.eval_score = 0


class CoFiTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel = None,
            args: TrainingArguments = None,
            additional_args: AdditionalArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            l0_module=None,
            teacher_model=None,
            **kwargs,
    ):

        Trainer.__init__(self, model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

        self.additional_args = additional_args

        self.l0_module = l0_module
        self.prepruning_finetune_steps = 100
        self.start_prune = False

        self.l0_optimizer = None
        self.lagrangian_optimizer = None

        self.eval_counter = Eval_Counter()
        self.start_saving_best = True if self.additional_args.pruning_type is None else False

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.args.device)

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logger.setLevel(log_level)

    def create_optimizer_and_scheduler(self, num_training_steps: int, build_l0_optimizer:bool=True):
        def log_params(param_groups, des):
            for i, grouped_parameters in enumerate(param_groups):
                logger.info(
                    f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            freeze_keywords = ["embeddings"]

            main_model_params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
            ]
            log_params(main_model_params, "main params")
            self.optimizer = AdamW(
                main_model_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

            if build_l0_optimizer and self.l0_module is not None:
                l0_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" not in n],
                    "weight_decay": 0.0,
                    "lr": self.additional_args.reg_learning_rate
                }]
                log_params(l0_params, "l0 reg params")
                self.l0_optimizer = AdamW(l0_params,
                                          betas=(self.args.adam_beta1,
                                                 self.args.adam_beta2),
                                          eps=self.args.adam_epsilon, )

                lagrangian_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                    "weight_decay": 0.0,
                    "lr": -self.additional_args.reg_learning_rate
                }]
                log_params(lagrangian_params, "l0 reg lagrangian params")
                self.lagrangian_optimizer = AdamW(lagrangian_params,
                                                    betas=(self.args.adam_beta1,
                                                            self.args.adam_beta2),
                                                    eps=self.args.adam_epsilon)

        if self.lr_scheduler is None:
            if self.additional_args.scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = None

    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(
            train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1) #! 12272

        if self.l0_module is not None:
            lagrangian_warmup_steps = self.additional_args.lagrangian_warmup_epochs * num_update_steps_per_epoch #! 24544
            # self.prepruning_finetune_steps = self.additional_args.prepruning_finetune_epochs * num_update_steps_per_epoch
            self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
            logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
            logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            self.t_total = int(num_update_steps_per_epoch *
                               self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = self.t_total

        self.create_optimizer_and_scheduler(num_training_steps=self.t_total, build_l0_optimizer = self.start_prune)

        model = self.model

        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")

        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d",
                    self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0

        epochs_trained = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        reg_loss = torch.tensor(0.0).to(self.args.device)
        lag_loss = torch.tensor(0.0).to(self.args.device)

        logging_loss_scalar = 0.0
        logging_reg_loss_scalar = 0.0
        logging_lag_loss_scalar = 0.0

        model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()

        self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(
            np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)

        self.evaluate()

        # training
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))): #! 20 epoch
            epoch_start = time.time()

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration",
                              disable=disable_tqdm)
            self.eval_counter.clear()

            for step, inputs in enumerate(epoch_iterator):
                if self.prepruning_finetune_steps > 0 and self.global_step == self.prepruning_finetune_steps: #! before pruning, run 12272 steps
                    self.start_prune = True

                    self.optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.t_total - self.global_step

                    # reset the optimizer
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
                    logger.info("Starting l0 regularization!")

                if self.start_prune:
                    zs = self.l0_module.forward(training=True) #! get the zs
                    self.fill_inputs_with_zs(zs, inputs) #! use the zs

                loss_terms = self.training_step(model, inputs)
                tr_loss_step = loss_terms["loss"]
                lag_loss_step = loss_terms["lagrangian_loss"]

                tr_loss += tr_loss_step
                lag_loss += lag_loss_step if lag_loss_step is not None else 0.0

                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()

                    if self.l0_module is not None and self.l0_optimizer is not None:
                        self.l0_optimizer.step()
                        self.lagrangian_optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self.l0_module is not None:
                        self.l0_module.constrain_parameters()

                    model.zero_grad()
                    if self.l0_module is not None:
                        self.l0_module.zero_grad()
                    self.optimizer.zero_grad()
                    if self.l0_optimizer is not None:
                        self.l0_optimizer.zero_grad()
                    if self.lagrangian_optimizer is not None:
                        self.lagrangian_optimizer.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        reg_loss_scalar = reg_loss.item()
                        lag_loss_scalar = lag_loss.item()

                        logs["loss"] = (
                            tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["reg_loss"] = (
                            reg_loss_scalar - logging_reg_loss_scalar) / self.args.logging_steps
                        logs["lag_loss"] = (
                            lag_loss_scalar - logging_lag_loss_scalar) / self.args.logging_steps

                        # backward compatibility for pytorch schedulers
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler.get_last_lr()[0] if version.parse(
                                torch.__version__) >= version.parse("1.4") else self.lr_scheduler.get_lr()[0]
                        else:
                            lr = self.args.learning_rate

                        logs["learning_rate"] = lr
                        logging_loss_scalar = tr_loss_scalar
                        logging_reg_loss_scalar = reg_loss_scalar
                        logging_lag_loss_scalar = lag_loss_scalar

                        self.log(logs)

                    if self.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                epoch_pbar.update(1)

                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

            epoch_end = time.time()
            # wandb.log({'epoch':epoch})
            logger.info(
                f"Epoch {epoch} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")

            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        # wandb.log({'global_step':self.global_step,'training_loss':tr_loss.item() / self.global_step})
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step, None)

    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None) -> PredictionOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # disable output hidden states and attention during evaluation
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False

        model = self.model

        # multi-gpu eval
        model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm

        zs = None
        if self.start_prune:
            self.l0_module.eval()
            zs = self.l0_module.forward(training=False)

        if zs is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(zs)

        for ii, inputs in enumerate(tqdm(dataloader, desc=description, disable=disable_tqdm)):
            if zs is not None:
                if ii == 0:
                    logger.info(f"Putting zs {zs.keys()} into inputs:")
                self.fill_inputs_with_zs(zs, inputs) #! use the zs
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only)

            batch_size = inputs[list(inputs.keys())[0]].shape[0]

            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels)
            if loss is not None:
                if type(loss) == float:
                    losses = [loss] * batch_size
                    if losses_host is None:
                        losses_host = losses
                    else:
                        losses_host.extend(losses)
                else:
                    losses = loss.repeat(batch_size)
                    losses_host = losses if losses_host is None else torch.cat(
                        (losses_host, losses), dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation.py loop
            delattr(self, "_past")

        if losses_host is not None:
            if not torch.is_tensor(losses_host):
                losses_host = torch.tensor(losses_host)
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(
                predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None and len(all_losses) > 0:
            metrics["eval_loss"] = np.mean(all_losses)

        if zs is not None:
            lag_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)

            expected_sparsity = round(expected_sparsity.item(), 5)
            metrics.update(pruned_model_size_info)
            metrics["expected_sparsity"] = expected_sparsity
            metrics["target_sparsity"] = target_sparsity

            if (not self.start_saving_best) and (expected_sparsity - self.additional_args.target_sparsity >= -self.additional_args.sparsity_epsilon):
                self.start_saving_best = True
                logger.info(f"Starting saving the best from epoch {int(self.epoch)} and step {self.global_step}")

        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True

        return PredictionOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[Dict[str, float], List]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        # wandb.log(output.metrics)
        output.metrics["step"] = self.global_step

        logger.info(f"Evaluating: {output.metrics}")

        eval_score = 0

        name = glue_tasks[self.model.config.finetuning_task]
        if isinstance(name, str):
            if name in output.metrics:
                eval_score = output.metrics[name]
        else:
            for na in name:
                if na in output.metrics:
                    eval_score = output.metrics[na]
                    break

        # logger.info(f"starting saving best: {self.global_step} {self.start_saving_best}")

        if self.start_saving_best:
            best_so_far = self.eval_counter.update(
                self.epoch, self.global_step, eval_score)
            if best_so_far:
                best_dir = os.path.join(self.args.output_dir, "best")
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)

                if self.l0_module is not None:
                    zs = self.l0_module.forward(training=False)
                    torch.save(zs, os.path.join(best_dir, "zs.pt"))
                    torch.save(self.l0_module, os.path.join(
                        best_dir, "l0_module.pt"))
                logger.info(f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.global_step} | Model size: {output.metrics['remaining_params'] if 'remaining_params' in output.metrics else 'Full' } | Score: {round(eval_score, 5)}]")
                self.model.save_pretrained(best_dir)

        return output.metrics

    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))

        zs = self.l0_module.forward(training=False)
        torch.save(zs, os.path.join(output_dir, "zs.pt"))

        self.model.save_pretrained(output_dir)

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        if self.additional_args.do_layer_distill: #! only do layer distill
            mlp_z = None
            head_layer_z = None
            if "mlp_z" in zs:
                mlp_z = zs["mlp_z"].detach().cpu()
            if "head_layer_z" in zs:
                head_layer_z = zs["head_layer_z"].detach().cpu()

            teacher_layer_output = teacher_outputs[2][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768]
            student_layer_output = student_outputs[2][1:] 

            # distilliting existing layers
            if self.additional_args.layer_distill_version == 2:
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                    s_layer_o = self.model.layer_transformation(s_layer_o)
                    l = mse_loss(t_layer_o, s_layer_o)
                    if mlp_z[layer_num] > 0:
                        layer_loss += l

            # distilling layers with a minimal distance
            elif self.additional_args.layer_distill_version > 2:
                l = []
                specified_teacher_layers = [2, 5, 8, 11]
                transformed_s_layer_o = [self.model.layer_transformation(
                    s_layer_o) for s_layer_o in student_layer_output]
                specified_teacher_layer_reps = [
                    teacher_layer_output[i] for i in specified_teacher_layers] #! teacher: 4x[32,113,768]

                device = transformed_s_layer_o[0].device
                for t_layer_o in specified_teacher_layer_reps:
                    for i, s_layer_o in enumerate(transformed_s_layer_o): #! student: 12x[32,113,768]
                        l.append(mse_loss(t_layer_o, s_layer_o))
                layerwiseloss = torch.stack(l).reshape(
                    len(specified_teacher_layer_reps), len(student_layer_output)) #! [4,12]

                existing_layers = None
                if head_layer_z is not None:
                    existing_layers = head_layer_z != 0
                    existing_layers = existing_layers.to(layerwiseloss.device)

                layer_loss = 0
                #! no ordering restriction specified
                if self.additional_args.layer_distill_version == 3:
                    alignment = torch.argmin(layerwiseloss, dim=1)
                #! added the ordering restriction -> to choose the min loss in 4 student layers
                elif self.additional_args.layer_distill_version == 4:
                    last_aligned_layer = 12
                    alignment = []
                    for search_index in range(3, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        if existing_layers is not None:
                            align = indexes[(
                                indexes < last_aligned_layer) & existing_layers]
                        else:
                            align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    alignment = torch.tensor(alignment).to(device)
                else:
                    logger.info(
                        f"{self.additional_args.layer_distill_version} version is not specified.")
                    sys.exit()

                layerwise = torch.arange(4).to(device)
                layer_loss += layerwiseloss[layerwise, alignment].sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
                if self.global_step % 100 == 0:
                    logger.info(f"v{self.additional_args.layer_distill_version} Global step: {self.global_step}, Alignment: " + str(alignment))
            return layer_loss
        else:
            return None

    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs):
        layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs)
        distill_loss = layer_loss

        ce_distill_loss = F.kl_div(
            input=F.log_softmax(
                student_outputs[1] / self.additional_args.distill_temp, dim=-1), #! logits: [32,3]
            target=F.softmax(
                teacher_outputs[1] / self.additional_args.distill_temp, dim=-1), #! distill_temp: 2.0
            reduction="batchmean") * (self.additional_args.distill_temp ** 2)

        loss = self.additional_args.distill_ce_loss_alpha * ce_distill_loss
        if distill_loss is not None:
            loss += self.additional_args.distill_loss_alpha * distill_loss

        return distill_loss, ce_distill_loss, loss

    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]


    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()
        if self.l0_module is not None:
            self.l0_module.train()
        inputs = self._prepare_inputs(inputs)

        distill_loss = None
        distill_ce_loss = None
        if self.teacher_model is not None:
            with torch.no_grad():
                # only retain inputs of certain keys
                teacher_inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
                                       "output_attentions", "output_hidden_states", "return_dict"]
                teacher_inputs = {key: inputs[key]
                                  for key in teacher_inputs_keys if key in inputs}
                self.shortens_inputs(teacher_inputs)
                teacher_outputs = self.teacher_model(**teacher_inputs)
            self.shortens_inputs(inputs)
            student_outputs = model(**inputs) #! get the two outputs

            zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs
            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                teacher_outputs, student_outputs, zs)
        else:
            loss = self.compute_loss(model, inputs)

        lagrangian_loss = None
        if self.start_prune:
            lagrangian_loss, _, _ = \
                self.l0_module.lagrangian_regularization(
                    self.global_step - self.prepruning_finetune_steps)
            loss += lagrangian_loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        
        # wandb.log({"loss": loss.detach(),
        #         "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
        #         "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
        #         "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None})
        
        return {"loss": loss.detach(),
                "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
                "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
                "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None}

    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
