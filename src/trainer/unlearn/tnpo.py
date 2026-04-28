import torch

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll, compute_tnpo_loss


class TNPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_tnpo_loss(
            model=model,
            ref_model=self.ref_model,
            inputs=forget_inputs,
            beta=self.beta,
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


class DrTNPO(TNPO):
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

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_tnpo_loss(
            model=model,
            ref_model=self.ref_model,
            inputs=forget_inputs,
            beta=self.beta,
        )

        if self.forget_dro:
            if self.log_ori_loss:
                self.log({"forget_loss_ori": forget_loss.clone().detach().mean().item()})
            forget_loss = -self.beta_dv_forget * torch.log(
                torch.mean(torch.exp(-forget_loss / self.beta_dv_forget))
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

        assert (
            self.retain_loss_type == "NLL"
        ), "DRO only supports NLL retain loss currently."
        retain_loss, _ = compute_batch_nll(model, retain_inputs)
        if self.retain_dro:
            if self.log_ori_loss:
                self.log({"retain_loss_ori": retain_loss.clone().detach().mean().item()})
            retain_loss = -self.beta_dv_retain * torch.log(
                torch.mean(torch.exp(-retain_loss / self.beta_dv_retain))
            )
        else:
            retain_loss = retain_loss.mean()
        self.log({"retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
