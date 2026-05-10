import torch

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll, compute_klmin_loss


class KLMin(GradDiff):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_klmin_loss(
            model=model,
            ref_model=self.ref_model,
            inputs=forget_inputs,
        )
        forget_loss = forget_loss.mean()

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


class DrKLMin(KLMin):
    def __init__(
        self,
        beta_dv_forget,
        beta_dv_retain,
        forget_dro=True,
        retain_dro=False,
        log_ori_loss=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta_dv_forget = beta_dv_forget
        self.beta_dv_retain = beta_dv_retain
        self.forget_dro = forget_dro
        self.retain_dro = retain_dro
        self.log_ori_loss = log_ori_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_klmin_loss(
            model=model,
            ref_model=self.ref_model,
            inputs=forget_inputs,
        )

        if self.forget_dro:
            if self.log_ori_loss:
                self.log({"forget_loss_ori": forget_loss.clone().detach().mean().item()})
            # Soft-max over per-sequence -KL losses: upweights hard-to-forget samples
            # (those with small KL = seq_loss closer to 0). Uses +β·log(E[exp(L/β)])
            # rather than the DrTNPO -β·log(E[exp(-L/β)]) because KLMin's sign
            # convention is inverted: harder = less negative, not more negative.
            forget_loss = self.beta_dv_forget * torch.log(
                torch.mean(torch.exp(forget_loss / self.beta_dv_forget))
            )
        else:
            forget_loss = forget_loss.mean()
        self.log({"forget_loss": forget_loss.item()})

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        if self.retain_dro:
            assert (
                self.retain_loss_type == "NLL"
            ), "DRO only supports NLL retain loss currently."
            retain_loss, _ = compute_batch_nll(model, retain_inputs)
            if self.log_ori_loss:
                self.log({"retain_loss_ori": retain_loss.clone().detach().mean().item()})
            retain_loss = -self.beta_dv_retain * torch.log(
                torch.mean(torch.exp(-retain_loss / self.beta_dv_retain))
            )
        else:
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        self.log({"retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
