# ===========================================================================
# _sam.py  —  HUST-BCIML EEG-decoding benchmark

# Original authors:    Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur (2021) — "Sharpness-Aware Minimization for Efficiently Improving Generalization", Proc. ICLR
#                      Original code: https://github.com/davda54/sam (canonical PyTorch implementation)
# Implementation:      David Samuel (davda54) — davda54/sam (https://github.com/davda54/sam)
# Current code:        Siyang Li — sylyoung/DeepTransferEEG (https://github.com/sylyoung/DeepTransferEEG) (used by FedBS; ported from)
# ===========================================================================
"""Sharpness-Aware Minimization optimizer (Foret et al., ICLR 2021).

Two-pass update: ``first_step`` ascends to the worst-case parameter perturbation
in an L2 ball of radius ``rho``, ``second_step`` restores the weights and applies
the base optimizer's step using the gradient taken at the perturbed point — so
the descent targets a *flat* minimum. SAR (Niu et al., ICLR 2023) uses it for
robust test-time entropy minimization.

Vendored verbatim from DeepTransferEEG ``tl/models/sam.py`` (the widely used
davda54/sam implementation). Prefixed with ``_`` so the registry auto-scan skips
it — it is a helper, not a Strategy plug-in.
"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Restore every parameter that was perturbed, keyed on whether ``old_p``
        # exists rather than on whether it has a gradient *this* pass. A parameter
        # that had a first-pass gradient (so was climbed to ``w + e(w)``) but no
        # second-pass gradient — conditional execution, a masked branch, a frozen
        # sub-module — would otherwise be left sitting at the adversarial point and
        # carry the perturbation into every later update.
        for group in self.param_groups:
            for p in group["params"]:
                old = self.state.get(p, {}).pop("old_p", None)
                if old is not None:
                    p.data = old                 # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
