import datasets
from transformers.trainer_utils import PredictionOutput
from trainer.trainer import CoFiTrainer 
import os
import torch
import torch.nn.functional as F
import logging

logger = logging.get_logger(__name__)

class CoFiQATrainer(CoFiTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        CoFiTrainer.__init__(self, *args, **kwargs)
        
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None):

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                # ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)
            self.log(metrics)
        else:
            metrics = {}
        metrics.update(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self.log(metrics)
        metrics["global_step"] = self.global_step

        self.logger.info(f"Evaluating: {metrics}")
        name = "f1"
        eval_score = metrics[name]

        if self.start_saving_best:
            best_so_far = self.eval_counter.update(
                self.epoch, self.global_step, eval_score)

            if best_so_far:
                best_dir = os.path.join(self.args.output_dir, "best")
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)
                zs = None

                torch.save(zs, os.path.join(best_dir, "zs.pt"))
                torch.save(self.l0_module, os.path.join(
                    best_dir, "l0_module.pt"))
                logger.info(f"Saving the best model so far: [Epoch {self.epoch} | Step: {self.global_step} | \
                                Model size: {output.metrics['remaining_params']} | Score: {eval_score}]")
                self.model.save_pretrained(best_dir)
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        eval_preds = self.post_process_function(test_examples, test_dataset, output.predictions)
        metrics = self.compute_metrics(eval_preds)
        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)

    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs):
        layer_distill_loss = self.calculate_layer_distillation_loss(zs)
        distill_ce_loss = F.kl_div(
            input=F.log_softmax(student_outputs.start_logits / self.additional_args.distill_temp, dim=-1),
            target=F.softmax(teacher_outputs.start_logits / self.additional_args.distill_temp, dim=-1),
            reduction="batchmean",
        ) * (self.additional_args.distill_temp ** 2)

        distill_ce_loss += F.kl_div(
            input=F.log_softmax(student_outputs.end_logits / self.additional_args.distill_temp, dim=-1),
            target=F.softmax(teacher_outputs.end_logits / self.additional_args.distill_temp, dim=-1),
            reduction="batchmean",
        ) * (self.additional_args.distill_temp ** 2)

        distill_ce_loss = distill_ce_loss / 2

        loss = self.additional_args.distill_loss_alpha * layer_distill_loss + self.additional_args.distill_ce_loss_alpha * distill_ce_loss
        return layer_distill_loss, distill_ce_loss, loss
