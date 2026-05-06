import copy
from types import SimpleNamespace

import torch

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_kl_divergence


class CPURefModel:
    """Reference model kept on CPU to free ~16 GB of GPU memory for 8B+ models.

    All callers use it inside torch.no_grad(), so CPU-side inference is safe.
    Inputs are moved to CPU, outputs (logits only) are moved back to the
    original device so downstream GPU computations work unchanged.
    """

    def __init__(self, model):
        self._model = model.eval()

    def __call__(self, **kwargs):
        target_device = next(
            (v.device for v in kwargs.values() if isinstance(v, torch.Tensor)),
            None,
        )
        cpu_inputs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        with torch.inference_mode():
            out = self._model(**cpu_inputs)
        logits = out.logits.to(target_device) if target_device is not None else out.logits
        return SimpleNamespace(logits=logits)

    def eval(self):
        return self


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        if self.is_deepspeed_enabled:
            ref_model = copy.deepcopy(model).to(self.accelerator.device)
            ref_model.eval()
            return self._prepare_deepspeed(ref_model)
        # CPU offload: move model to CPU first so deepcopy never peaks at 2x GPU.
        # Frees ~16 GB on a 40 GB GPU, making 8B model training feasible.
        device = next(model.parameters()).device
        model.cpu()
        cpu_ref = copy.deepcopy(model)
        model.to(device)
        torch.cuda.empty_cache()
        return CPURefModel(cpu_ref)

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        self.log({"forget_loss": forget_loss.item(), "retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
