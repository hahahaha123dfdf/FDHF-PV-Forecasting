"""Low-frequency PV forecasting with a reprogrammed GPT-2 (Time-LLM, Jin et al.,
ICLR 2024, https://github.com/KimMeen/Time-LLM), as described in manuscript
Section 3.2.

The frozen GPT-2-Small backbone is kept fixed; only the patch embedding, the
cross-modal reprogramming layer, the vocabulary mapping layer and the output
head are trained. The reprogramming source is the mapped LLM vocabulary
embeddings, and the PaP natural-language prefix (built in prompt_bank.py) is
prepended to the LLM input ahead of the reprogrammed time-series tokens. On top
of the reference design, the only additions are the scikit-learn-style
fit/predict wrapper and the few-shot plumbing used by the FDHF pipeline.

Requires ``transformers``; GPT-2 weights are downloaded on first use.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

from .prompt_bank import build_pap_prompt, compute_window_stats


# --------------------------------------------------------------------------- #
# Backbone resolution
# --------------------------------------------------------------------------- #
def _local_snapshot_candidates(name: str):
    """Local directories that may already hold the pretrained backbone."""
    cands = []
    env_dir = os.environ.get("FDHF_BACKBONE_DIR")
    if env_dir:
        cands.append(os.path.join(env_dir, name))
    hub = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    cands += [
        os.path.join(hub, "hub", name),
        os.path.join(hub, name),
    ]
    cands += sorted(glob.glob(os.path.join(
        hub, "hub", "models--" + name.replace("/", "--"), "snapshots", "*")))
    # some installations keep a single backbone directly in the cache root
    cands += [os.path.join(hub, "hub"), hub]
    return cands


def _snapshot_is_usable(path: str, model_cls) -> bool:
    """A local directory can serve as a backbone snapshot only if it holds the
    files the given class needs."""
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    if issubclass(model_cls, GPT2Tokenizer):
        return bool(files & {"vocab.json", "tokenizer.json", "merges.txt"})
    return "config.json" in files


def load_backbone(model_cls, name: str):
    """Load ``model_cls.from_pretrained(name)``, falling back to a local
    snapshot when the Hugging Face hub is unreachable.

    Keeps the pipeline runnable on an offline workstation that already has the
    backbone in its local model cache.
    """
    try:
        return model_cls.from_pretrained(name)
    except Exception as hub_error:
        for cand in _local_snapshot_candidates(name):
            if _snapshot_is_usable(cand, model_cls):
                try:
                    return model_cls.from_pretrained(cand)
                except Exception:
                    continue
        raise hub_error


# --------------------------------------------------------------------------- #
# Patch embedding and cross-modal reprogramming layer.
# --------------------------------------------------------------------------- #
class ReplicationPad1d(nn.Module):
    """Replicate the last value of the time axis (upstream Time-LLM)."""

    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        replicate = x[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        return torch.cat([x, replicate], dim=-1)


class TokenEmbedding(nn.Module):
    """Circular-padded 1-D convolution that projects a patch to d_model
    dimensions (upstream Time-LLM)."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in",
                                nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, T) -> patches (B*V, n_patch, patch_len) -> (B*V, n_patch, d_model)
        B, V, T = x.shape
        x = self.padding_patch_layer(x)
        patches = x.unfold(2, self.patch_len, self.stride)   # (B, V, n_patch, patch_len)
        patches = patches.reshape(B * V, -1, self.patch_len)
        return self.dropout(self.value_embedding(patches))


class ReprogrammingLayer(nn.Module):
    """Cross-attention that reprograms source (patch) tokens into the frozen
    LLM's embedding space, using the *mapped LLM vocabulary embeddings* as
    keys/values (identical to upstream Time-LLM, where the reprogramming source
    is the vocabulary, NOT the PaP text)."""

    def __init__(self, d_model: int, n_heads: int, d_keys: int = None,
                 d_llm: int = None, attention_dropout: float = 0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape          # S = num_tokens (no batch dim)
        H = self.n_heads
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,she->blhe", A, value_embedding)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class TimeLLMModel(nn.Module):
    def __init__(self, llm_dim: int = 768, patch_len: int = 16, stride: int = 8,
                 n_heads: int = 4, pred_len: int = 24, input_len: int = 96,
                 num_tokens: int = 1000):
        super().__init__()
        self.llm_dim = llm_dim
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred_len
        # number of patches actually produced by the padded unfolding
        self.n_patch = (input_len + stride - patch_len) // stride + 1

        self.llm = load_backbone(GPT2Model, "gpt2")
        for p in self.llm.parameters():
            p.requires_grad = False
        self.word_embeddings = self.llm.get_input_embeddings()
        self.vocab_size = self.word_embeddings.weight.shape[0]
        # mapping_layer: vocab_size -> num_tokens
        self.mapping_layer = nn.Linear(self.vocab_size, num_tokens)
        self.num_tokens = num_tokens

        self.patch_embedding = PatchEmbedding(patch_len, stride, llm_dim)
        self.reprogramming = ReprogrammingLayer(llm_dim, n_heads, llm_dim, llm_dim)
        self.head = nn.Linear(self.n_patch * llm_dim, pred_len)
        self.dropout = nn.Dropout(0.1)

    def _source_embeddings(self) -> torch.Tensor:
        # mapped LLM vocabulary embeddings -> (num_tokens, d_llm)
        return self.mapping_layer(
            self.word_embeddings.weight.permute(1, 0)
        ).permute(1, 0)

    def forward(self, x: torch.Tensor, prompt_ids: torch.Tensor,
                prompt_mask: torch.Tensor | None = None):
        # x: (B, V, T), prompt_ids: (B, L), prompt_mask: (B, L) with 1 = real token
        B, V, T = x.shape
        x_enc = self.patch_embedding(x)                 # (B*V, n_patch, d_model)
        n_patch = x_enc.shape[1]

        prompt_emb = self.word_embeddings(prompt_ids)   # (B, L, d_llm)  -- PaP PREFIX only
        prompt_emb = (prompt_emb.unsqueeze(1)
                      .expand(-1, V, -1, -1)
                      .reshape(B * V, -1, self.llm_dim))

        source_emb = self._source_embeddings()          # (num_tokens, d_llm) -- reprogramming source
        x_enc = self.reprogramming(x_enc, source_emb, source_emb)   # (B*V, n_patch, d_llm)

        llm_input = torch.cat([prompt_emb, x_enc], dim=1)        # (B*V, L + n_patch, d)

        # Padding positions of the PaP prefix must not be attended to. The
        # patch tokens are always valid, so the mask is [prompt_mask | ones].
        attn = None
        if prompt_mask is not None:
            pm = prompt_mask.unsqueeze(1).expand(-1, V, -1).reshape(B * V, -1)
            attn = torch.cat([pm, torch.ones(B * V, n_patch, dtype=pm.dtype,
                                             device=pm.device)], dim=1)
        dec = self.llm(inputs_embeds=llm_input, attention_mask=attn).last_hidden_state
        dec = dec[:, prompt_emb.shape[1]:]               # keep patch tokens
        dec = dec.reshape(B * V, n_patch * self.llm_dim)
        out = self.head(dec)                             # (B*V, pred_len)
        out = out.reshape(B, V, self.pred_len).mean(dim=1)   # (B, pred_len)
        return out


# --------------------------------------------------------------------------- #
# Scikit-learn-style forecaster wrapper
# --------------------------------------------------------------------------- #
class TimeLLMForecaster:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        lf = cfg["low_freq_branch"]
        self.patch_len = lf["patch_len"]
        self.stride = lf["patch_stride"]
        self.pred_len = cfg["data"]["pred_len"]
        self.input_len = cfg["data"]["input_len"]
        self.epochs = lf["max_epochs"]
        self.batch_size = lf["batch_size"]
        self.lr = lf["learning_rate"]
        self.device = cfg["training"]["device"]
        self.use_pap = lf["use_pap"]

        self.tokenizer = load_backbone(GPT2Tokenizer, "gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = TimeLLMModel(
            llm_dim=lf["llm_dim"], patch_len=self.patch_len,
            stride=self.stride, pred_len=self.pred_len, input_len=self.input_len,
        ).to(self.device)

    def _make_prompt_ids(self, X_np: np.ndarray):
        """Return (input_ids, attention_mask) for the PaP prefix."""
        if self.use_pap:
            prompts = []
            for i in range(X_np.shape[0]):
                stats = compute_window_stats(X_np[i])
                prompts.append(build_pap_prompt("pv", stats))
        else:
            prompts = ["Photovoltaic power forecasting."] * X_np.shape[0]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, V, input_len), y: (N, pred_len)."""
        import time
        t0 = time.perf_counter()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.model.train()
        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.cfg["low_freq_branch"]["weight_decay"],
        )
        loss_fn = nn.MSELoss()
        N = X.shape[0]
        for epoch in range(self.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb, yb = X[idx], y[idx]
                pids, pmask = self._make_prompt_ids(xb.cpu().numpy())
                pred = self.model(xb, pids, pmask)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.last_train_seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        import time
        t0 = time.perf_counter()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        outs = []
        N = X.shape[0]
        for s in range(0, N, self.batch_size):
            xb = X[s:s + self.batch_size]
            pids, pmask = self._make_prompt_ids(xb.cpu().numpy())
            outs.append(self.model(xb, pids, pmask).cpu().numpy())
        self.last_inference_seconds = time.perf_counter() - t0
        self.last_inference_samples = N
        return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------- #
# BERT backbone (manuscript Case 4: GPT-2-Small -> BERT)
# --------------------------------------------------------------------------- #
class BertForecaster:
    """Drop-in BERT (encoder-only) replacement for the low-frequency branch.

    Uses a frozen BERT base model instead of GPT-2 (manuscript Case 4). BERT is
    encoder-only, so the patch representations are mean-pooled and passed through
    the output head (no autoregressive decoding). The cross-modal reprogramming
    source is still the mapped LLM vocabulary embeddings (same as GPT-2 branch).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        lf = cfg["low_freq_branch"]
        self.patch_len = lf["patch_len"]
        self.stride = lf["patch_stride"]
        self.pred_len = cfg["data"]["pred_len"]
        self.input_len = cfg["data"]["input_len"]
        self.epochs = lf["max_epochs"]
        self.batch_size = lf["batch_size"]
        self.lr = lf["learning_rate"]
        self.device = cfg["training"]["device"]
        self.use_pap = lf["use_pap"]

        try:
            from transformers import BertModel, BertTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Case 4 (BERT backbone) requires the `transformers` package and "
                "`bert-base-uncased` weights. Install transformers and ensure "
                "network access, or remove Case 4 from the ablation study. "
                f"Import failed: {exc}"
            )
        self.tokenizer = load_backbone(BertTokenizer, "bert-base-uncased")
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.sep_token
        self.model = BertLowModel(
            llm_dim=768, patch_len=self.patch_len, stride=self.stride,
            pred_len=self.pred_len, input_len=self.input_len,
        ).to(self.device)

    def _make_prompt_ids(self, X_np: np.ndarray):
        """Return (input_ids, attention_mask) for the PaP prefix."""
        if self.use_pap:
            prompts = [build_pap_prompt("pv", compute_window_stats(X_np[i]))
                       for i in range(X_np.shape[0])]
        else:
            prompts = ["Photovoltaic power forecasting."] * X_np.shape[0]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=2048)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def fit(self, X, y):
        import time
        t0 = time.perf_counter()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.model.train()
        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.cfg["low_freq_branch"]["weight_decay"])
        loss_fn = nn.MSELoss()
        N = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, self.batch_size):
                idx = perm[s:s + self.batch_size]
                pids, pmask = self._make_prompt_ids(X[idx].cpu().numpy())
                pred = self.model(X[idx], pids, pmask)
                loss = loss_fn(pred, y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.last_train_seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        import time
        t0 = time.perf_counter()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        outs = []
        N = X.shape[0]
        for s in range(0, N, self.batch_size):
            xb = X[s:s + self.batch_size]
            pids, pmask = self._make_prompt_ids(xb.cpu().numpy())
            outs.append(self.model(xb, pids, pmask).cpu().numpy())
        self.last_inference_seconds = time.perf_counter() - t0
        self.last_inference_samples = N
        return np.concatenate(outs, axis=0)


class BertLowModel(nn.Module):
    def __init__(self, llm_dim=768, patch_len=16, stride=8, pred_len=24,
                 input_len=96, num_tokens=1000):
        super().__init__()
        from transformers import BertModel
        self.bert = load_backbone(BertModel, "bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.word_embeddings = self.bert.get_input_embeddings()
        self.vocab_size = self.word_embeddings.weight.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, num_tokens)
        self.patch_embedding = PatchEmbedding(patch_len, stride, llm_dim)
        self.reprogramming = ReprogrammingLayer(llm_dim, 4, llm_dim, llm_dim)
        self.head = nn.Linear(llm_dim, pred_len)
        self.dropout = nn.Dropout(0.1)

    def _source_embeddings(self):
        return self.mapping_layer(
            self.word_embeddings.weight.permute(1, 0)
        ).permute(1, 0)

    def forward(self, x, prompt_ids=None, prompt_mask=None):
        # The PaP prefix is consumed by the reprogramming source only; the
        # encoder input consists of the reprogrammed patch tokens, so no
        # padding mask is required here.
        B, V, T = x.shape
        x_enc = self.patch_embedding(x)                  # (B*V, n_patch, d)
        source_emb = self._source_embeddings()           # (num_tokens, d_llm)
        x_enc = self.reprogramming(x_enc, source_emb, source_emb)
        out = self.bert(inputs_embeds=x_enc).last_hidden_state
        out = out.mean(dim=1)            # pool over patch tokens (encoder-only)
        out = self.dropout(out)
        out = self.head(out)             # (B*V, pred_len)
        return out.reshape(B, V, self.pred_len).mean(dim=1)
