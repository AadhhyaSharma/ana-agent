#!/usr/bin/env python3
"""
neural_engine.py
================
Neural Engine for ANA Agent.
Covers: Weight Initialization, Optimizers, Loss Functions, Gradient Clipping,
        Learning Rate Schedulers, Regularization, Data Pipeline,
        Model Checkpointing, Training Loop, Evaluation, Visualization.
"""

import os, sys, json, math, time, random, hashlib, copy, itertools, functools
import datetime, threading
from collections import defaultdict, deque, OrderedDict
from typing import List, Dict, Tuple, Optional, Any, Callable

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TENSOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class T:
    """Pure-Python tensor operations (1D and 2D)."""

    @staticmethod
    def dot(a, b): return sum(x*y for x,y in zip(a,b))

    @staticmethod
    def add(a, b):
        if isinstance(a[0], list): return [[a[i][j]+b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
        return [x+y for x,y in zip(a,b)]

    @staticmethod
    def sub(a, b):
        if isinstance(a[0], list): return [[a[i][j]-b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
        return [x-y for x,y in zip(a,b)]

    @staticmethod
    def mul(a, s):
        if isinstance(a[0], list): return [[v*s for v in row] for row in a]
        return [x*s for x in a]

    @staticmethod
    def div(a, b):
        if isinstance(b, (int, float)): return T.mul(a, 1.0/b)
        return [x/y for x,y in zip(a,b)]

    @staticmethod
    def matmul(A, B):
        rows, cols, inner = len(A), len(B[0]), len(B)
        return [[sum(A[i][k]*B[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]

    @staticmethod
    def transpose(A): return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

    @staticmethod
    def outer(a, b): return [[ai*bi for bi in b] for ai in a]

    @staticmethod
    def norm(a):
        if isinstance(a[0], list): return math.sqrt(sum(v**2 for row in a for v in row)+1e-12)
        return math.sqrt(sum(x*x for x in a)+1e-12)

    @staticmethod
    def normalize(a): n=T.norm(a); return [x/n for x in a]

    @staticmethod
    def zeros(n):
        if isinstance(n, tuple): return [[0.0]*n[1] for _ in range(n[0])]
        return [0.0]*n

    @staticmethod
    def ones(n):
        if isinstance(n, tuple): return [[1.0]*n[1] for _ in range(n[0])]
        return [1.0]*n

    @staticmethod
    def rand(n, std=0.1):
        def rn(): u1=random.random()+1e-12; u2=random.random(); return std*math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
        if isinstance(n, tuple): return [[rn() for _ in range(n[1])] for _ in range(n[0])]
        return [rn() for _ in range(n)]

    @staticmethod
    def clip_vec(v, lo, hi): return [max(lo, min(hi, x)) for x in v]

    @staticmethod
    def clip_mat(A, lo, hi): return [[max(lo, min(hi, v)) for v in row] for row in A]

    @staticmethod
    def sum_vec(v): return sum(v)

    @staticmethod
    def mean(v): return sum(v)/max(1, len(v))

    @staticmethod
    def var(v): m=T.mean(v); return sum((x-m)**2 for x in v)/max(1, len(v))

    @staticmethod
    def std(v): return math.sqrt(T.var(v)+1e-8)

    @staticmethod
    def abs_vec(v): return [abs(x) for x in v]

    @staticmethod
    def sqrt_vec(v): return [math.sqrt(abs(x)+1e-12) for x in v]

    @staticmethod
    def pow_vec(v, p): return [x**p for x in v]

    @staticmethod
    def log_vec(v): return [math.log(abs(x)+1e-12) for x in v]

    @staticmethod
    def exp_vec(v): return [math.exp(min(x, 500)) for x in v]

    @staticmethod
    def flatten(A):
        if not A or not isinstance(A[0], list): return A
        return [v for row in A for v in row]

    @staticmethod
    def reshape(v, rows, cols):
        assert len(v) == rows*cols
        return [v[i*cols:(i+1)*cols] for i in range(rows)]

    @staticmethod
    def concat(a, b): return a + b

    @staticmethod
    def stack(vecs): return vecs[:]

    @staticmethod
    def cosine_sim(a, b): return T.dot(a,b)/(T.norm(a)*T.norm(b)+1e-12)

    @staticmethod
    def l1_norm(v): return sum(abs(x) for x in v)

    @staticmethod
    def l2_norm(v): return math.sqrt(sum(x*x for x in v))

    @staticmethod
    def softmax(v):
        m=max(v); e=[math.exp(min(x-m, 500)) for x in v]; s=sum(e)+1e-12
        return [x/s for x in e]

    @staticmethod
    def log_softmax(v):
        m=max(v); log_sum=math.log(sum(math.exp(min(x-m,500)) for x in v)+1e-12)+m
        return [x-log_sum for x in v]

    @staticmethod
    def argmax(v): return v.index(max(v))

    @staticmethod
    def argmin(v): return v.index(min(v))

    @staticmethod
    def topk(v, k): return sorted(range(len(v)), key=lambda i:-v[i])[:k]

    @staticmethod
    def pad(v, size, val=0.0): return (v+[val]*size)[:size]

    @staticmethod
    def batch_norm_vec(v, gamma=None, beta=None, eps=1e-5):
        m=T.mean(v); s=T.std(v)
        norm=[(x-m)/max(s,eps) for x in v]
        if gamma is not None: norm=[g*n for g,n in zip(gamma,norm)]
        if beta is not None:  norm=[n+b for n,b in zip(norm,beta)]
        return norm


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WEIGHT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class Init:
    """Weight initialization strategies: Xavier, He, LeCun, Orthogonal, Sparse."""

    @staticmethod
    def xavier_uniform(n_in, n_out):
        lim = math.sqrt(6.0/(n_in+n_out))
        return [[random.uniform(-lim, lim) for _ in range(n_out)] for _ in range(n_in)]

    @staticmethod
    def xavier_normal(n_in, n_out):
        std = math.sqrt(2.0/(n_in+n_out))
        return T.rand((n_in, n_out), std)

    @staticmethod
    def he_uniform(n_in, n_out):
        lim = math.sqrt(6.0/n_in)
        return [[random.uniform(-lim, lim) for _ in range(n_out)] for _ in range(n_in)]

    @staticmethod
    def he_normal(n_in, n_out):
        std = math.sqrt(2.0/n_in)
        return T.rand((n_in, n_out), std)

    @staticmethod
    def lecun_uniform(n_in, n_out):
        lim = math.sqrt(3.0/n_in)
        return [[random.uniform(-lim, lim) for _ in range(n_out)] for _ in range(n_in)]

    @staticmethod
    def lecun_normal(n_in, n_out):
        std = math.sqrt(1.0/n_in)
        return T.rand((n_in, n_out), std)

    @staticmethod
    def orthogonal(n, gain=1.0):
        """Approximate orthogonal init via QR decomposition (pure Python)."""
        A = T.rand((n, n), 1.0)
        Q = []
        for i in range(n):
            v = A[i][:]
            for u in Q:
                proj = T.dot(v, u)
                v = T.sub(v, T.mul(u, proj))
            nrm = T.norm(v)
            if nrm > 1e-10:
                Q.append(T.mul(v, 1.0/nrm))
            else:
                Q.append(T.zeros(n))
        return [[gain*Q[i][j] for j in range(n)] for i in range(n)]

    @staticmethod
    def sparse(n_in, n_out, sparsity=0.9):
        std = math.sqrt(2.0/n_in)
        W = T.rand((n_in, n_out), std)
        for i in range(n_in):
            for j in range(n_out):
                if random.random() < sparsity:
                    W[i][j] = 0.0
        return W

    @staticmethod
    def zeros(n_in, n_out): return T.zeros((n_in, n_out))

    @staticmethod
    def ones(n_in, n_out): return T.ones((n_in, n_out))

    @staticmethod
    def constant(n_in, n_out, val): return [[val]*n_out for _ in range(n_in)]

    @staticmethod
    def uniform(n_in, n_out, lo=-0.1, hi=0.1):
        return [[random.uniform(lo, hi) for _ in range(n_out)] for _ in range(n_in)]

    @staticmethod
    def auto(n_in, n_out, activation="relu"):
        """Pick best init strategy for given activation."""
        if activation in ("relu", "leaky_relu"): return Init.he_normal(n_in, n_out)
        elif activation in ("sigmoid", "tanh"):  return Init.xavier_normal(n_in, n_out)
        elif activation == "gelu":               return Init.he_normal(n_in, n_out)
        elif activation == "softmax":            return Init.xavier_uniform(n_in, n_out)
        else:                                    return Init.lecun_normal(n_in, n_out)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class Loss:
    """Loss functions for training."""

    @staticmethod
    def mse(pred, target):
        """Mean Squared Error."""
        return T.mean([T.pow_vec(T.sub([p], [t]), 2) for p, t in zip(pred, target)])

    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error."""
        return T.mean(T.abs_vec(T.sub(pred, target)))

    @staticmethod
    def cross_entropy(pred, target):
        """Cross Entropy Loss."""
        return -T.sum_vec([t*math.log(p+1e-12) for t, p in zip(target, pred)])

    @staticmethod
    def binary_cross_entropy(pred, target):
        """Binary Cross Entropy."""
        return -T.mean([t*math.log(p+1e-12) + (1-t)*math.log(1-p+1e-12) for t, p in zip(target, pred)])

    @staticmethod
    def huber(pred, target, delta=1.0):
        """Huber Loss."""
        errors = T.abs_vec(T.sub(pred, target))
        return T.mean([0.5*e**2 if e <= delta else delta*(e - 0.5*delta) for e in errors])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════════════════

class Optimizer:
    """Base optimizer class."""
    def __init__(self, lr=0.001):
        self.lr = lr
        self.step_count = 0

    def update(self, params, grads):
        raise NotImplementedError

    def step(self):
        self.step_count += 1


class SGD(Optimizer):
    """Stochastic Gradient Descent."""
    def __init__(self, lr=0.001, momentum=0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.velocities = None

    def update(self, params, grads):
        if self.velocities is None:
            self.velocities = [[0.0]*len(p) if isinstance(p, list) else 0.0 for p in params]
        
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if isinstance(p, list):
                v = self.velocities[i]
                v = [self.momentum*v[j] - self.lr*g[j] for j in range(len(p))]
                self.velocities[i] = v
                updated.append([p[j] + v[j] for j in range(len(p))])
            else:
                v = self.momentum*self.velocities[i] - self.lr*g
                self.velocities[i] = v
                updated.append(p + v)
        
        self.step()
        return updated


class Adam(Optimizer):
    """Adam Optimizer."""
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(lr)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [[0.0]*len(p) if isinstance(p, list) else 0.0 for p in params]
            self.v = [[0.0]*len(p) if isinstance(p, list) else 0.0 for p in params]
        
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if isinstance(p, list):
                m_i = self.m[i]
                v_i = self.v[i]
                m_i = [self.b1*m_i[j] + (1-self.b1)*g[j] for j in range(len(p))]
                v_i = [self.b2*v_i[j] + (1-self.b2)*g[j]**2 for j in range(len(p))]
                self.m[i] = m_i
                self.v[i] = v_i
                m_hat = [m_i[j]/(1-self.b1**(self.step_count+1)) for j in range(len(p))]
                v_hat = [v_i[j]/(1-self.b2**(self.step_count+1)) for j in range(len(p))]
                updated.append([p[j] - self.lr*m_hat[j]/(math.sqrt(v_hat[j])+self.eps) for j in range(len(p))])
            else:
                m_i = self.b1*self.m[i] + (1-self.b1)*g
                v_i = self.b2*self.v[i] + (1-self.b2)*g**2
                self.m[i] = m_i
                self.v[i] = v_i
                m_hat = m_i/(1-self.b1**(self.step_count+1))
                v_hat = v_i/(1-self.b2**(self.step_count+1))
                updated.append(p - self.lr*m_hat/(math.sqrt(v_hat)+self.eps))
        
        self.step()
        return updated


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LEARNING RATE SCHEDULERS
# ═══════════════════════════════════════════════════════════════════════════════

class LRScheduler:
    """Learning rate scheduler."""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    """Step decay learning rate scheduler."""
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        if self.optimizer.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(LRScheduler):
    """Exponential decay learning rate scheduler."""
    def __init__(self, optimizer, gamma=0.99):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler."""
    def __init__(self, optimizer, T_max=100, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self):
        t = self.optimizer.step_count % self.T_max
        self.optimizer.lr = self.eta_min + 0.5*(self.optimizer.lr - self.eta_min)*(1 + math.cos(math.pi*t/self.T_max))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class TrainConfig:
    """Training configuration."""
    def __init__(self, epochs=10, batch_size=32, lr=0.001, optimizer="adam", loss="mse"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.loss = loss
        self.early_stopping = False
        self.patience = 5
        self.validation_split = 0.2

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "validation_split": self.validation_split
        }


class Checkpoint:
    """Model checkpoint for saving/loading."""
    def __init__(self, model_state, optimizer_state, epoch, loss):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.epoch = epoch
        self.loss = loss
        self.timestamp = time.time()

    def save(self, filepath):
        """Save checkpoint to file."""
        data = {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "epoch": self.epoch,
            "loss": self.loss,
            "timestamp": self.timestamp
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load checkpoint from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(data["model_state"], data["optimizer_state"], data["epoch"], data["loss"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """Generic training loop."""
    def __init__(self, model, config: TrainConfig):
        self.model = model
        self.config = config
        self.history = {"loss": [], "val_loss": []}
        self.best_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_data):
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_data:
            batch_loss = self.model.train_batch(batch)
            total_loss += batch_loss
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        self.history["loss"].append(avg_loss)
        return avg_loss

    def validate(self, val_data):
        """Validate on validation data."""
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_data:
            batch_loss = self.model.train_batch(batch)
            total_loss += batch_loss
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

    def train(self, train_data, val_data=None):
        """Full training loop."""
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_data)
            
            if val_data:
                val_loss = self.validate(val_data)
                
                if self.config.early_stopping:
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {train_loss:.4f}")

    def get_history(self):
        """Get training history."""
        return self.history
