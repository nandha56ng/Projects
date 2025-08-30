
"""
Implementation of FAAD-Mem:
Feature-Adapted Attention-Driven Analytic Learning with Memory Augmentation and Knowledge Distillation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict, Any, Sequence
from .AnalyticLinear import RecursiveLinear
from .Learner import Learner, loader_t
from utils import validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from tqdm import tqdm
import copy
import time
import os
import math

class MemoryBank:
    def __init__(self, feature_dim: int, max_per_class: int = 1):
        self.max_per_class = max_per_class
        self.storage: Dict[int, list[torch.Tensor]] = {}
        self.feature_dim = feature_dim

    def add(self, features: torch.Tensor, labels: torch.Tensor):
        for feat, label in zip(features, labels):
            label = label.item()
            if label not in self.storage:
                self.storage[label] = []
            if len(self.storage[label]) < self.max_per_class:
                self.storage[label].append(feat.detach().cpu())

    def get_memory(self):
        if not self.storage:
            return None, None
        max_class = max(self.storage.keys()) + 1
        X, Y = [], []
        for class_id, feats in self.storage.items():
            proto = torch.stack(feats).mean(dim=0)  # prototype averaging
            X.append(proto)
            y = torch.zeros(max_class)
            y[class_id] = 1.0
            Y.append(y)
        return torch.stack(X), torch.stack(Y)


class FeatureAdaptation(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim , out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x): 
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        alpha = self.fc(x)
        return x * alpha


class FAADMem(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_output_dim: int, adapt_dim: int, num_classes: int,
                 gamma: float = 1e-3, kd_temp: float = 2.0, lambda_kd: float = 0.5, l2_reg_alpha = 0.0, device=None):
        super().__init__()
        self.backbone = backbone
        self.adapt = FeatureAdaptation(backbone_output_dim, adapt_dim).to(device)
        self.attn = AttentionModule(adapt_dim).to(device)
        self.linear = RecursiveLinear(adapt_dim, gamma, device=device, dtype=torch.double)
        self.memory = MemoryBank(adapt_dim, max_per_class=1)
        self.kd_temp = kd_temp
        self.lambda_kd = lambda_kd
        self.l2_reg_alpha = l2_reg_alpha
        self.teacher = None
        self.ema_momentum = 0.99
        self.device = device
        self.scheduler = None
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.current_phase = 0
        self.total_phases = 1
        self.bias = nn.Parameter(torch.zeros(num_classes))


    def forward(self, x):
        with torch.no_grad():
            x_feat = self.backbone(x)
            if len(x_feat.shape) == 4:
                x_feat = F.adaptive_avg_pool2d(x_feat, (1, 1))
                x_feat = torch.flatten(x_feat, 1) 

        x_adapt = self.adapt(x_feat)
        x_attn = self.attn(x_adapt)
        x_attn = F.normalize(x_attn, p=2, dim=1)
        logits = self.linear(x_attn)
        if logits.shape[1] != self.bias.shape[0]:
            return logits, x_attn
        return logits + self.bias, x_attn



    @torch.no_grad()
    def fit(self, x, y):
        logits, features = self.forward(x)
        num_classes = max(self.linear.out_features, int(y.max().item()) + 1)
        if logits.shape[1] < num_classes:
            logits = F.pad(logits, (0, num_classes - logits.shape[1]))
        Y_cur = F.one_hot(y, num_classes=num_classes).double()

        X_mem, Y_mem = self.memory.get_memory()
        if X_mem is not None and Y_mem is not None:
            X_total = torch.cat([features, X_mem.to(x.device)], dim=0)
            if Y_mem.shape[1] < Y_cur.shape[1]:
                Y_mem = F.pad(Y_mem, (0, Y_cur.shape[1] - Y_mem.shape[1]))
            Y_total = torch.cat([Y_cur, Y_mem.to(x.device)], dim=0)
        else:
            X_total = features
            Y_total = Y_cur

        self.linear.fit(X_total, Y_total)
        self.linear.update()
        self.memory.add(features, y)

        # Feature regularization
        l2_reg = torch.norm(features, p=2) ** 2

        # Dynamic lambda adjustment based on current phase
        lambda_kd_scaled = 0.3 * (1 - math.cos(math.pi * self.current_phase / self.total_phases))


        ce_loss = self.ce_loss_fn(logits, y)
        kd_loss = self.compute_kd_loss(logits, x)
        total_loss = (1 - lambda_kd_scaled) * ce_loss + lambda_kd_scaled * kd_loss + self.l2_reg_alpha * l2_reg

        for _ in range(3):
            X_replay, Y_replay = self.memory.get_memory()
            if X_replay is not None and Y_replay is not None:
                self.linear.fit(X_replay.to(x.device), Y_replay.to(x.device))
                self.linear.update()

    def update_teacher(self):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self)

            self.teacher.eval()
        else:
            # EMA update
            for param_t, param_s in zip(self.teacher.parameters(), self.parameters()):
                if param_t.data.shape != param_s.data.shape:
                    continue
                param_t.data = self.ema_momentum * param_t.data + (1.0 - self.ema_momentum) * param_s.data

    def compute_kd_loss(self, logits, x):
        if self.teacher is None:
            return torch.tensor(0.0, device=logits.device)
        
        with torch.no_grad():
            teacher_logits, _ = self.teacher(x)

        known_classes = logits.shape[1]
        teacher_logits = teacher_logits[:, :known_classes]
        if teacher_logits.shape[1] < logits.shape[1]:
            teacher_logits = F.pad(teacher_logits, (0, logits.shape[1] - teacher_logits.shape[1]))
        elif teacher_logits.shape[1] > logits.shape[1]:
            logits = F.pad(logits, (0, teacher_logits.shape[1] - logits.shape[1]))


        p= F.log_softmax(logits / self.kd_temp, dim=1)
        q = F.softmax(teacher_logits / self.kd_temp, dim=1)

        if p.shape[1] < q.shape[1]:
            p = F.pad(p, (0, q.shape[1] - p.shape[1]))
        elif p.shape[1] > q.shape[1]:
            q = F.pad(q, (0, p.shape[1] - q.shape[1]))

        return F.kl_div(p, q, reduction='batchmean') * (self.kd_temp ** 2)

#    def benchmark_inference(self, dataloader, device='cuda'):
#        self.eval()
#        self.to(device)
#        total_time = 0.0
#        total_samples = 0
#        with torch.no_grad():
#            for batch, _ in dataloader:
#                batch = batch.to(device)
#                start = time.time()
#                _ = self.forward(batch)
#                total_time += (time.time() - start)
#                total_samples += batch.size(0)
#        avg_time = (total_time / total_samples) * 1000  # in ms
#        throughput = total_samples / total_time         # img/sec
#        print(f"Inference Time: {avg_time:.3f} ms/image | Throughput: {throughput:.2f} img/sec")

class FAADMemLearner(Learner):
    def __init__(
            self, 
            args: Dict[str, Any], 
            backbone: nn.Module, 
            backbone_output: int, 
            device=None,
            all_devices: Optional[Sequence[DeviceLikeType]] = None
            ):
        super().__init__(args, backbone, backbone_output, device, all_devices)
        self.adapt_dim = args.get("adapt_dim", 2048)
        self.gamma = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
        self.model = FAADMem(
            backbone, 
            backbone_output, 
            self.adapt_dim, args["phases"] * args["IL_batch_size"],
            gamma=self.gamma,
            kd_temp=self.args.get("kd_temp", 2.0),
            lambda_kd=self.args.get("lambda_kd", 0.5),
            l2_reg_alpha=self.args.get("l2_reg_alpha", 0.0),
            device=device)

    def base_training(
            self,
            train_loader: loader_t, 
            val_loader: loader_t, 
            baseset_size: int,
            ) -> None:
        model = torch.nn.Sequential(
            self.backbone,
            nn.Linear(self.backbone_output, baseset_size),
        ).to(self.device)

        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args["learning_rate"], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.base_epochs - self.warmup_epochs, eta_min=1e-6 # type: ignore
        )
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], [self.warmup_epochs]
            )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        logging_file_path = os.path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(1, self.args["base_epochs"] + 1):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                    f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
                )
            model.train()
            for X, y in tqdm(train_loader, "Training"):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
        # Optional: validate on base phase data
            train_meter = validate(
                model, train_loader, baseset_size, desc="Training (Validation)")
            print(
                f"loss: {train_meter.loss:.4f}",
                f"acc@1: {train_meter.accuracy * 100:.3f}%",
                f"acc@5: {train_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {train_meter.f1_micro * 100:.3f}%",
                sep="    ",
                )
            val_meter = validate(model, val_loader, baseset_size, desc="Validation")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                if epoch != 0:
                    self.save_object(
                        (self.backbone, X.shape[1], self.backbone_output),
                        "backbone.pth",
                        )
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"acc@5: {val_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
                )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.accuracy5,
                val_meter.f1_micro,
                train_meter.loss,
                train_meter.accuracy,
                train_meter.accuracy5,
                train_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
                )
        logging_file.close()

        self.backbone.load_state_dict(model[0].state_dict())
        self.backbone.eval()
        self.model.backbone.load_state_dict(self.backbone.state_dict())
#        self.model.benchmark_inference(val_loader, device=self.device)

    def learn(self, data_loader: loader_t, incremental_size: int, desc: str = "Incremental Learning") -> None:
        self.model.eval()
        for epoch in range(self.args.get("IL_epochs", 1)):
            print(f"Incremental Epoch {epoch + 1}/{self.args.get('IL_epochs', 1)}")
            for X, y in tqdm(data_loader, desc=f"{desc} Epoch {epoch + 1}"):
                X, y = X.to(self.device), y.to(self.device)
                self.model.fit(X, y)
        self.model.update_teacher()

    def before_validation(self) -> None:
        pass  # Nothing to update explicitly, teacher already set after learning

    def inference(self, X: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(X)
        return logits