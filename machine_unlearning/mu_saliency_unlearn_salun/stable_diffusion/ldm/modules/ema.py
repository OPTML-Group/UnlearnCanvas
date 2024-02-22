import torch
from torch import nn


class LitEma(nn.Module):  # LitEma is an implementation of Exponential Moving Average (EMA) for PyTorch models. EMA is a common technique used in many machine learning models, especially in optimization algorithms, to get a smoothed, average value of parameters over time. This can be helpful to stabilize the learning process and to avoid overfitting.
    def __init__(self, model, decay=0.9999, use_num_upates=True, handle_non_trainable=False):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.frozen_param_names = set(name for name, p in model.named_parameters() if not p.requires_grad)
        self.handle_non_trainable = handle_non_trainable
        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))  # defines the rate at which the importance of older observations decreases
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
                             else torch.tensor(-1, dtype=torch.int))  # whether or not to use the number of updates in the decay calculation
        # store a clone of the model's parameters that will be used to hold the EMA parameters.
        for name, p in model.named_parameters():
            if p.requires_grad or handle_non_trainable:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def update_frozen_params(self, model):
        self.frozen_param_names = set(name for name, p in model.named_parameters() if not p.requires_grad)

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                # Check if this parameter is frozen
                if self.handle_non_trainable and key in self.frozen_param_names:
                    continue  # Skip EMA update for frozen parameter
                # CHANGED: Added a condition to handle non-trainable parameters
                if m_param[key].requires_grad or (self.handle_non_trainable and not m_param[key].requires_grad):
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):  # copies the current EMA parameters to the model parameters.
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            # CHANGED: Added a condition to handle non-trainable parameters
            if m_param[key].requires_grad or (self.handle_non_trainable and not m_param[key].requires_grad):
            # if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name, print(f"keys {key} not found in shadow parameters")

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
