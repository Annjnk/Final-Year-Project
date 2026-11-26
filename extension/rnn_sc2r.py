"""GRU, LSTM and BiLSTM sequence classifiers for the LEAP extension, section D.

Copied/adapted from:
    Repository: https://github.com/lengocluyen/semantics-constrained-counterfactuel-recourse (SC2R)
    File:       src/cbe_cf/models/train.py, function train_rnn_sequence
    Commit:     7bd85230dcb099e94885cdbc2373780778c6bd39 (7bd8523)

Kept exactly as in train_rnn_sequence:
    - Recurrent layer: nn.GRU or nn.LSTM, hidden 64, 1 layer, batch_first, dropout 0 (their 0.1 only
      applies with more than one layer). BiLSTM is the LSTM with bidirectional=True.
    - Head: the last time step's output, out[:, -1, :], concatenated with the static features,
      then nn.LayerNorm and nn.Linear(..., 1).
    - Standardisation: per feature, mean and population std, std <= 1e-8 replaced by 1. The sequence
      statistics are fitted on the training rows flattened over time steps, the static ones per column.
      Their _standardize_fit helper is copied unchanged.
    - Training: AdamW (lr 1e-3, weight_decay 1e-4), BCEWithLogitsLoss with no class weighting, training
      batches of 128 shuffled, validation batches of 256, 30 epochs with no early stopping. The epoch with
      the lowest validation loss (mean of the per-batch losses, as theirs) is restored at the end.
    - Probabilities: 1 / (1 + exp(-logit)) on the logits, computed in NumPy as theirs.

Changes versus train_rnn_sequence:
    1. Packaged as a class with fit(seq, static, y) and predict_proba(seq, static). Their function reads
       Parquet files, scores the test split itself and writes reports, calibration JSON and a checkpoint.
       This class does no file input or output; the LEAP harness scores the predictions.
    2. Inputs come from the caller. Their sequence is 3 segments derived from cumulative *_w7, *_w14 and
       *_w28 snapshot columns, and their static inputs are the other numeric columns. In section D the
       sequence is one step for all pre-course activity plus one step per week up to the cut-off day,
       with total clicks, active days and unique resources per step, and the static inputs are LEAP's
       11 features.
    3. Validation split: 15% of the training rows, stratified on y with random_state=42, made inside fit.
       Theirs uses a separate set of course presentations listed in a splits file. The test rows are
       never passed to fit, so they play no part in training, scaling or choosing the best epoch.
    4. Standardisation statistics are fitted on the remaining 85% training rows, which excludes the
       validation rows exactly as theirs excludes its validation presentations.
    5. Seeds: random.seed, np.random.seed and torch.manual_seed with the seed given to the class, which is
       the LEAP split seed of each run (0 to 4) rather than their default 1337. Their set_global_seed also
       sets PYTHONHASHSEED, which has no effect once Python is running, so it is left out.
    6. Device is chosen by the caller ("cpu" or "mps"). Theirs uses CUDA when available, otherwise CPU.
    7. Only "gru", "lstm" and "bilstm" are offered. Their plain "rnn" option is left out.
    8. predict_proba returns two columns, P(y = 0) and P(y = 1), like scikit-learn. Theirs keeps one.
    9. The per-epoch training and validation losses are kept in history_ and the restored epoch in
       best_epoch_; theirs only prints them when verbose.

The SC2R repository is released under the MIT licence, reproduced here as it requires:

    MIT License
    
    Copyright (c) 2026
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ARCHS = ("gru", "lstm", "bilstm")
SC2R_COMMIT = "7bd85230dcb099e94885cdbc2373780778c6bd39"


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Copied unchanged from SC2R's train.py.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


class SeqClassifier:
    """Recurrent sequence model plus static features, trained as in SC2R's train_rnn_sequence."""

    def __init__(self, arch: str, seed: int = 42, device: str = "cpu", epochs: int = 30,
                 val_fraction: float = 0.15, val_random_state: int = 42) -> None:
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
        self.arch = arch
        self.seed = seed
        self.device = device
        self.epochs = epochs
        self.val_fraction = val_fraction
        self.val_random_state = val_random_state

    def _apply(self, seq_arr: np.ndarray, st_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq_out = (seq_arr - self.seq_mean_.reshape(1, 1, -1)) / self.seq_std_.reshape(1, 1, -1)
        if st_arr.shape[1]:
            st_out = (st_arr - self.static_mean_.reshape(1, -1)) / self.static_std_.reshape(1, -1)
        else:
            st_out = st_arr
        return seq_out.astype(np.float32), st_out.astype(np.float32)

    def _forward(self, seq_b: torch.Tensor, st_b: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn_(seq_b)
        last = out[:, -1, :]
        if st_b.shape[1]:
            last = torch.cat([last, st_b], dim=1)
        return self.head_(last).squeeze(1)

    def fit(self, seq, static, y) -> "SeqClassifier":
        seq = np.asarray(seq, dtype=np.float32)
        static = np.asarray(static, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if seq.ndim != 3 or static.ndim != 2 or not (len(seq) == len(static) == len(y)):
            raise ValueError(f"expected seq [N, T, F], static [N, S] and y [N]; got {seq.shape}, {static.shape}, {y.shape}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        device = torch.device(self.device)

        # Validation: 15% of the training rows, stratified, fixed random_state. Test rows never reach fit.
        rows = np.arange(len(y))
        train_rows, val_rows = train_test_split(
            rows, test_size=self.val_fraction, random_state=self.val_random_state, stratify=y
        )
        seq_train, st_train, y_train = seq[train_rows], static[train_rows], y[train_rows]
        seq_val, st_val, y_val = seq[val_rows], static[val_rows], y[val_rows]
        self.n_train_rows_, self.n_val_rows_ = len(train_rows), len(val_rows)

        # Standardise per feature with statistics from the training rows only.
        self.seq_mean_, self.seq_std_ = _standardize_fit(seq_train.reshape(-1, seq_train.shape[-1]))
        self.static_mean_, self.static_std_ = (
            _standardize_fit(st_train) if st_train.shape[1] else (np.zeros((0,), np.float32), np.ones((0,), np.float32))
        )
        seq_train, st_train = self._apply(seq_train, st_train)
        seq_val, st_val = self._apply(seq_val, st_val)

        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(seq_train), torch.from_numpy(st_train), torch.from_numpy(y_train)),
            batch_size=128,
            shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(torch.from_numpy(seq_val), torch.from_numpy(st_val), torch.from_numpy(y_val)),
            batch_size=256,
            shuffle=False,
        )

        hidden = 64
        num_layers = 1
        dropout = 0.1
        bidirectional = self.arch == "bilstm"
        if self.arch in {"lstm", "bilstm"}:
            rnn: nn.Module = nn.LSTM(
                input_size=seq_train.shape[-1],
                hidden_size=hidden,
                num_layers=num_layers,
                dropout=0.0 if num_layers == 1 else dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            rnn = nn.GRU(
                input_size=seq_train.shape[-1],
                hidden_size=hidden,
                num_layers=num_layers,
                dropout=0.0 if num_layers == 1 else dropout,
                batch_first=True,
                bidirectional=False,
            )
        out_dim = hidden * (2 if bidirectional else 1)
        head_in = out_dim + int(st_train.shape[1])
        head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, 1))
        self.rnn_ = rnn.to(device)
        self.head_ = head.to(device)

        params = list(self.rnn_.parameters()) + list(self.head_.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        best_state = None
        self.best_epoch_ = None
        self.history_ = []
        for epoch in range(int(self.epochs)):
            self.rnn_.train()
            self.head_.train()
            train_losses = []
            for seq_b, st_b, y_b in train_dl:
                seq_b, st_b, y_b = seq_b.to(device), st_b.to(device), y_b.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self._forward(seq_b, st_b), y_b)
                loss.backward()
                opt.step()
                train_losses.append(float(loss.detach().cpu().item()))
            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            self.rnn_.eval()
            self.head_.eval()
            losses = []
            with torch.no_grad():
                for seq_b, st_b, y_b in val_dl:
                    seq_b, st_b, y_b = seq_b.to(device), st_b.to(device), y_b.to(device)
                    losses.append(float(loss_fn(self._forward(seq_b, st_b), y_b).cpu().item()))
            val_loss = float(np.mean(losses)) if losses else float("inf")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    "rnn": {k: v.detach().cpu().clone() for k, v in self.rnn_.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in self.head_.state_dict().items()},
                }
                self.best_epoch_ = epoch + 1
            self.history_.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if best_state is not None:
            self.rnn_.load_state_dict(best_state["rnn"])
            self.head_.load_state_dict(best_state["head"])
        return self

    def predict_proba(self, seq, static) -> np.ndarray:
        seq_arr, st_arr = self._apply(np.asarray(seq, dtype=np.float32), np.asarray(static, dtype=np.float32))
        device = torch.device(self.device)
        self.rnn_.eval()
        self.head_.eval()
        with torch.no_grad():
            seq_b = torch.from_numpy(seq_arr).to(device)
            st_b = torch.from_numpy(st_arr).to(device)
            logits = self._forward(seq_b, st_b).detach().cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - proba, proba])
