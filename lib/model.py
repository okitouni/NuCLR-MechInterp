import torch
import math
from torch import nn
from torch.nn import functional as F
from functools import partial
import mup
import warnings
from typing import Callable, Iterable
from .data import Data
from .utils import inverse_transform
import os
import torch.nn.utils.parametrize as p


class Base(nn.Module):
    def __init__(
        self,
        vocab_size: Iterable,
        non_embedded_input_dim: int,
        hidden_dim: int,
        embedding_dim: int = None,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.non_embedded_input_dim = non_embedded_input_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim or hidden_dim
        self.share_embeddings = share_embeddings
        self.input_dim = self.embedding_dim * len(vocab_size) + non_embedded_input_dim
        # we are using weights here because that is more convenient for dimn_regularization
        if share_embeddings:
            self.emb = nn.Embedding(vocab_size[0], self.embedding_dim).weight
        else:
            self.emb = nn.ParameterList(
                [nn.Embedding(v, self.embedding_dim).weight for v in self.vocab_size]
            )
        self.hidden_dim = hidden_dim

    def embed_input(self, x, embs):
        if self.share_embeddings:
            embs = [embs[x[:, i].long()] for i, _ in enumerate(self.vocab_size)]
        else:
            embs = [embs[i][x[:, i].long()] for i, _ in enumerate(self.vocab_size)]
        if self.non_embedded_input_dim > 0:
            embs.append(x[:, len(self.vocab_size) :])
        return torch.cat(embs, dim=1)  # [ batch_size, 2 * hidden_dim ]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm: Callable = None,
    ):
        norm = norm or (lambda x: x)
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ff = nn.Sequential(
            norm(nn.Linear(d_model, d_model)),
            activation,
            norm(nn.Linear(d_model, d_model)),
            activation,
        )
        # self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # self.norm = nn.BatchNorm1d(d_model, affine=False)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.norm(x + self.dropout(self.ff(x)))



class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head=4, dropout=0.0):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class ResidualBlockWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm: Callable = None,
    ):
        norm = norm or (lambda x: x)
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.attn = norm(SelfAttention(d_model, n_head, dropout))
        self.ff = nn.Sequential(
            norm(nn.Linear(d_model, d_model)),
            activation,
        )
        # self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # self.norm = nn.BatchNorm1d(d_model, affine=False)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.attn(x)
        return self.norm(x + self.dropout(self.ff(x)))


class BaselineModel(Base):
    def __init__(
        self,
        vocab_size: Iterable,
        non_embedded_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        output_transform: Callable = None,
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__(vocab_size, non_embedded_input_dim, hidden_dim)
        act = nn.ReLU()
        self.nonlinear = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            *[
                ResidualBlock(hidden_dim, activation=act)
                for _ in range(depth)
            ],
        )
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.output_transform = output_transform

    def forward(self, x, return_shape='task'):
        return self.forward_with_embeddings(x, self.emb, return_shape)

    def forward_with_embeddings(self, x, embs, return_shape='task', apply_transform=True):  # embs: [ batch_size, 2 * hidden_dim ]
        task_idx = x[:, [2]]
        x = self.embed_input(x, embs)
        x = self.nonlinear(x)  # [ batch_size, hidden_dim ]
        x = self.readout(x)  # [ batch_size, output_dim ]
        if apply_transform and hasattr(self, "output_transform") and self.output_transform:
            x = self.output_transform(x)
        if return_shape == 'all':
            return x
        elif return_shape == 'task':
            x = x.gather(1, task_idx)
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")
        return x

class Transformer(Base):
    def __init__(
        self,
        vocab_size: Iterable,
        non_embedded_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        # n_head: int = 4,
        output_transform: Callable = None,
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__(vocab_size, non_embedded_input_dim, hidden_dim)
        act = nn.ReLU()
        self.nonlinear = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim),
            nn.SiLU(),
            *[
                ResidualBlockWithAttn(hidden_dim, n_head=4, activation=act)
                for _ in range(depth)
            ],
        )
        self.readout = nn.Linear(self.input_dim, output_dim)
        self.output_transform = output_transform

    def forward(self, x, return_shape='task'):
        return self.forward_with_embeddings(x, self.emb, return_shape)

    def forward_with_embeddings(self, x, embs, return_shape='task', apply_transform=True):  # embs: [ batch_size, 2 * hidden_dim ]
        task_idx = x[:, [2]]
        x = self.embed_input(x, embs)
        x = x.view(x.shape[0], -1, self.hidden_dim) 
        x = self.nonlinear(x)  # [ batch_size, T, hidden_dim ]
        x = x.view(x.shape[0], -1)
        x = self.readout(x)  # [ batch_size, output_dim ]
        if apply_transform and hasattr(self, "output_transform") and self.output_transform:
            x = self.output_transform(x)
        if return_shape == 'all':
            return x
        elif return_shape == 'task':
            x = x.gather(1, task_idx)
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")
        return x

    

def get_model_fn(config):
    if config.MODEL == "baseline":
        return BaselineModel
    elif config.MODEL == "transformer":
        return Transformer
    else:
        raise ValueError(
            f"Unknown model: {config.MODEL}, choose between 'baseline', 'splitup', 'transformer' and 'moe'"
        )


def _append_readout(model_fn: Callable) -> Callable:
    """Append a muP readout to a model. If the model is a sequential model,
    the readout replaces the last element in the sequence. Otherwise,
    the readout layer is expected to be an attribute.

    Args:
        model_fn (callable): Function which returns a model.
    """

    def model_fn_with_readout(*args, **kwargs):
        model = model_fn(*args, **kwargs)
        # check if model already has a readout, FIXME: this is a hack
        if any([isinstance(x, mup.MuReadout) for x in model.modules()]):
            return model
        if isinstance(model, nn.Sequential):
            assert isinstance(
                model[-1], nn.Linear
            ), "Last layer of sequential model must be linear (readout)"
            old_readout = model.pop(len(model) - 1)
            model.append(mup.MuReadout(*old_readout.weight.T.shape))
        else:
            assert hasattr(
                model, "readout"
            ), "Model must be sequential or have a readout attribute"
            old_readout = model.readout
            model.readout = mup.MuReadout(*old_readout.weight.T.shape)
        return model

    return model_fn_with_readout


def make_mup(model_fn: Callable, shape_file=None, model=None, **scale_kwargs) -> nn.Module:
    """Reinitialize model with mup scaling of relevant dimensions. Takes a function which returns a model and returns a model with mup scaling.
    Assumes the model has a readout linear layer which is either the last layer in a sequential model or an attribute of the model.

    Args:
        model_fn (Callable): Function which returns a nn.Module model.
        init_fn (Callable, optional): Function which initializes the model parameters in-place. Defaults to Kaiming uniform with a = sqrt(5).

    Raises:
        ValueError: If depth is in scale_kwargs. Depth is not a scaling parameter.

    Returns:
        nn.Module: Model with mup scaling.
    """
    if "depth" in (k.lower() for k in scale_kwargs.keys()):
        warnings.warn(
            "Depth found in scale_kwargs. Scaling depth is not allowed by muP. Is this intentional?"
        )
    model_fn = _append_readout(model_fn)
    base_kwargs = {k: 32 for k in scale_kwargs}
    delta_kwargs = {k: 64 for k in scale_kwargs}
    base = model_fn(**base_kwargs)
    delta = model_fn(**delta_kwargs)
    model = model_fn(**scale_kwargs) if model is None else model
    mup.set_base_shapes(model, base, delta=delta, savefile=shape_file, do_assert=False)
    del base, delta
    for name, param in model.named_parameters():
        if "weight" in name.lower() or "emb" in name.lower():  # FIXME or not
            # mup.init.uniform_(param, -.1, .1)
            mup.init.kaiming_uniform_(param, a=5**0.5, nonlinearity="leaky_relu")
    return model


def get_model_and_optim(data: Data, config, shape_file=None):
    """uses data to figure various shapes, config for model_params, and shape_file to save shapes"""
    # set up model
    if config.MODEL == "splitup":
        output_dim = list(data.output_map.values())
    else:
        output_dim = sum(data.output_map.values())

    model_fn = get_model_fn(config)
    model_fn = partial(
        model_fn,
        vocab_size=data.vocab_size,
        non_embedded_input_dim=data.X.shape[1] - len(data.vocab_size),
        output_dim=output_dim,
        depth=config.DEPTH,
        output_transform=partial(inverse_transform, data=data),
    )
    model = make_mup(model_fn, shape_file, hidden_dim=config.HIDDEN_DIM).to(config.DEV)
    # model = model_fn(hidden_dim=config.HIDDEN_DIM).to(config.DEV)
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "bias" in n.lower()]},
        {
            "params": [
                p for n, p in model.named_parameters() if "bias" not in n.lower()
            ],
            "weight_decay": config.WD,
        },
    ]
    # optimizer = mup.MuSGD(param_groups, lr=config.LR, momentum=.99, nesterov=True)
    if hasattr(config, "OPTIM") and config.OPTIM == "sgd":
        optimizer = mup.MuSGD(param_groups, lr=config.LR)
    else:
        optimizer = mup.MuAdamW(param_groups, lr=config.LR)
    # split into weights biases
    # optimizer = torch.optim.AdamW(param_groups, lr=config.LR, amsgrad=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WD)
    return model, optimizer


class AutoEncoder(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
    self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)
    self.input_bias = torch.nn.Parameter(torch.zeros(input_dim))

  def forward(self, x):
    x = self.encoder(x - self.input_bias)
    acts = torch.nn.functional.relu(x)
    x = self.decoder(acts) + self.input_bias
    return x, acts

  def loss(self, x, lambda_l1=.1):
    y, acts = self(x)
    return torch.nn.functional.mse_loss(y, x) + lambda_l1 * acts.abs().mean()
