# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Subset
from timm.utils import ModelEma
from timm.data import Mixup
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import balanced_accuracy_score
import math

from data_loaderv2 import get_dataloader, set_seed
from model400 import get_model

# ===== 基础配置 =====
DATA_ROOT   = "/home/node/zzz/data/train400"
SAVE_PATH   = "/home/node/zzz/model400o2u.pth"
NUM_CLASSES = 400

EPOCHS      = 50
BASE_BATCH_SIZE = 32      # 半监督阶段使用较小的batch_size
FINAL_BATCH_SIZE = 48     # 全监督阶段使用较大的batch_size
SEMI_START_EPOCH = 15
SEMI_END_EPOCH   = 20
FINAL_START_EPOCH = 21    # 从epoch

BASE_LR     = 1e-4
WEIGHT_DECAY= 6e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = (DEVICE.type == "cuda")

# Checkpoint 配置
CKPT_DIR    = "/home/node/zzz/checks400"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
RESUME_PATH = "/home/node/zzz/checks400/ckpt_latest.pth"

# ===== 增强 TTA 配置 =====
TTA_HFLIP = True  # 水平翻转
TTA_MULTI_SCALE = True  # 多尺度TTA
# 修正：基于320训练，使用稍小和稍大的尺度
TTA_SCALES = [304, 320, 336]  # 围绕320的多尺度

print(f"🖥 使用设备: {DEVICE.type} | AMP: {AMP_ENABLED}")
print(f"🎯 TTA配置: 水平翻转={TTA_HFLIP}, 多尺度={TTA_MULTI_SCALE}, 尺度={TTA_SCALES}")

# ===== AMP =====
from torch import amp
from torch.amp import GradScaler
def autocast_ctx(): return amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED)
scaler = GradScaler(device=('cuda' if DEVICE.type == 'cuda' else 'cpu'), enabled=AMP_ENABLED)

# ===== Mixup & Loss（逐样本）=====
MIXUP_ALPHA = 0.4  # 稍微增大Mixup强度
CUTMIX_ALPHA = 0.8
mixup_fn = Mixup(
    mixup_alpha=MIXUP_ALPHA,
    cutmix_alpha=CUTMIX_ALPHA,
    cutmix_minmax=(0.20, 0.60),
    label_smoothing=0.0,      # Mixup 路径不做 LS，避免双平滑
    num_classes=NUM_CLASSES
)

class SoftTargetCrossEntropyPerSample(nn.Module):
    def forward(self, logits, targets_soft):
        log_probs = F.log_softmax(logits, dim=1)
        return (-targets_soft * log_probs).sum(dim=1)

criterion_soft = SoftTargetCrossEntropyPerSample()
criterion_hard = nn.CrossEntropyLoss(label_smoothing=0.05, reduction='none')

# ===== O2U 配置（8–12）=====
O2U_ENABLE       = True
O2U_START_EPOCH  = 8
O2U_END_EPOCH    = 12
O2U_CYCLES       = 2
O2U_LR_MAX       = 1e-4
O2U_LR_MIN       = 3e-5
O2U_DROP_FRAC    = 0.05

# ===== 半监督改进配置 =====
TAU = 0.95
LAMBDA_U_MAX = 1.0  # 最大无监督权重
ADAPTIVE_RATIO = 0.85  # 自适应阈值比例

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr

def o2u_triangular_lr(epoch_in_window, window_len, cycles, lr_min, lr_max):
    cycle_len = window_len / cycles
    pos = (epoch_in_window % cycle_len) / max(1e-8, cycle_len)
    if pos <= 0.5:
        t = pos / 0.5
        return lr_max + (lr_min - lr_max) * t
    else:
        t = (pos - 0.5) / 0.5
        return lr_min + (lr_max - lr_min) * t

# ===== 改进1: 自适应阈值 =====
def adaptive_threshold(conf, ratio=ADAPTIVE_RATIO, max_tau=TAU):
    """
    自适应阈值：选择前ratio%最置信的样本，但不超过max_tau
    """
    if len(conf) == 0:
        return torch.zeros_like(conf)

    k = max(1, int(len(conf) * ratio))
    if k >= len(conf):
        return (conf >= max_tau).float()

    threshold = torch.topk(conf, k)[0][-1]
    threshold = min(threshold.item(), max_tau)
    return (conf >= threshold).float()

# ===== 改进2: 动态λ调度 =====
def get_lambda_u(epoch, semi_start, semi_end, max_lambda=LAMBDA_U_MAX):
    if epoch < semi_start:
        return 0.0
    total_semi_epochs = semi_end - semi_start
    current_progress = (epoch - semi_start) / total_semi_epochs
    if current_progress <= 0.2:
        ramp_up = 0.5 * (1 - math.cos(math.pi * current_progress / 0.2))
        return max_lambda * ramp_up
    else:
        return max_lambda

# ---- 增强 TTA：多尺度 + 水平翻转 ----
def infer_with_tta(model, x, hflip=False, multi_scale=False, scales=None):
    with torch.no_grad(), autocast_ctx():
        if multi_scale and scales:
            logits_list = []
            for scale in scales:
                # 调整输入尺寸
                if x.shape[-1] != scale or x.shape[-2] != scale:
                    x_scaled = F.interpolate(x, size=scale, mode='bilinear', align_corners=False)
                else:
                    x_scaled = x
                
                logits_scale = model(x_scaled)
                if hflip:
                    logits_flip = model(torch.flip(x_scaled, dims=[-1]))
                    logits_scale = (logits_scale + logits_flip) / 2
                logits_list.append(logits_scale)
            
            # 多尺度结果平均
            logits = torch.stack(logits_list).mean(0)
        else:
            # 单尺度推理
            logits = model(x)
            if hflip:
                logits_flip = model(torch.flip(x, dims=[-1]))
                logits = (logits + logits_flip) / 2
    
    return logits

# ===== 验证（使用增强 TTA）=====
def evaluate(model, dataloader):
    model.eval(); model.to(DEVICE)
    total, correct1, correct5 = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                imgs5, labels, _, _ = batch
                labels = labels.to(DEVICE)
                B, V, C, H, W = imgs5.shape
                imgs5 = imgs5.view(-1, C, H, W).to(DEVICE)
                # 使用增强TTA
                logits = infer_with_tta(model, imgs5, hflip=TTA_HFLIP, multi_scale=TTA_MULTI_SCALE, scales=TTA_SCALES).view(B, V, -1).mean(1)
            else:
                (img_g, img_2nd), labels, _, _, mask = batch
                labels = labels.to(DEVICE)
                imgs = img_g.to(DEVICE)
                # 使用增强TTA
                logits = infer_with_tta(model, imgs, hflip=TTA_HFLIP, multi_scale=TTA_MULTI_SCALE, scales=TTA_SCALES)
            total += labels.size(0)
            _, pred5 = logits.topk(5, dim=1, largest=True, sorted=True)
            eq = pred5.eq(labels.view(-1, 1))
            correct1 += eq[:, :1].sum().item()
            correct5 += eq.sum().item()
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    import torch as _t
    all_preds  = _t.cat(all_preds).numpy()
    all_labels = _t.cat(all_labels).numpy()
    top1 = 100. * correct1 / max(1, total)
    top5 = 100. * correct5 / max(1, total)
    bal  = 100. * balanced_accuracy_score(all_labels, all_preds)
    print(f" 验证 Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Balanced: {bal:.2f}%")
    return top1

# ===== 动态重建loader =====
def rebuild_loader_excluding(noisy_paths_set, batch_size):
    full_loader, _ = get_dataloader(DATA_ROOT, split="train", batch_size=batch_size, num_workers=8)
    ds = full_loader.dataset
    keep_indices = [i for i, p in enumerate(ds.img_paths) if p not in noisy_paths_set]
    subset = Subset(ds, keep_indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True
    )
    return loader, len(ds.img_paths), len(keep_indices)

def rebuild_train_loader(batch_size):
    train_loader, _ = get_dataloader(DATA_ROOT, split="train", batch_size=batch_size, num_workers=8)
    return train_loader

# ====== Checkpoint 辅助 ======
def save_ckpt(epoch, model, ema, optimizer, scaler, cosine, best_acc, o2u_scores, current_batch_size, suffix="latest"):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": (getattr(ema, 'ema', None) or getattr(ema, 'module', None) or model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cosine_state": cosine.state_dict(),
        "best_acc": best_acc,
        "o2u_scores": o2u_scores,
        "current_batch_size": current_batch_size,
    }
    path = os.path.join(CKPT_DIR, f"ckpt_{suffix}.pth")
    torch.save(ckpt, path)
    return path

def load_ckpt(path, model, ema, optimizer, scaler, cosine):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if hasattr(ema, "ema"):
        ema.ema.load_state_dict(ckpt["ema"])
    else:
        ema.module.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    cosine.load_state_dict(ckpt["cosine_state"])
    best_acc = ckpt.get("best_acc", 0.0)
    o2u_scores = ckpt.get("o2u_scores", {})
    start_epoch = ckpt["epoch"] + 1
    current_batch_size = ckpt.get("current_batch_size", BASE_BATCH_SIZE)
    print(f"🔁 已从 {path} 恢复，将从 epoch {start_epoch} 继续训练。")
    return start_epoch, best_acc, o2u_scores, current_batch_size

def train():
    set_seed(42)
    current_batch_size = FINAL_BATCH_SIZE

    train_loader = rebuild_train_loader(current_batch_size)
    val_loader, _ = get_dataloader(DATA_ROOT, split="val", batch_size=current_batch_size, num_workers=8)
    unlabeled_loader = rebuild_train_loader(BASE_BATCH_SIZE)
    unlabeled_iter = iter(unlabeled_loader)

    model = get_model(num_classes=NUM_CLASSES, pretrained=True, device=DEVICE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    # ===== 修正的学习率调度器（兼容旧版timm）=====
    cosine = CosineLRScheduler(
        optimizer, 
        t_initial=EPOCHS, 
        lr_min=3e-6,              # 更低的最终学习率，让优化更精细
        warmup_t=2, 
        warmup_lr_init=3e-5,
        warmup_prefix=True        # warmup阶段不计入总epoch
    )

    ema = ModelEma(model, decay=0.999)

    best_acc = 0.0
    o2u_scores = {}
    start_epoch = 0
    o2u_applied = False

    if RESUME_PATH and os.path.isfile(RESUME_PATH):
        print(f"🔄 强制从 {RESUME_PATH} 恢复训练...")
        start_epoch, best_acc, o2u_scores, current_batch_size = load_ckpt(RESUME_PATH, model, ema, optimizer, scaler, cosine)
        if start_epoch >= O2U_END_EPOCH and o2u_scores and not o2u_applied:
            print("🔧 检测到O2U已完成但剔除未应用，立即应用噪声剔除...")
            scored = sorted(o2u_scores.items(), key=lambda x: x[1], reverse=True)
            drop_k = int(len(scored) * O2U_DROP_FRAC)
            noisy_paths = set([p for p, _ in scored[:drop_k]])
            print(f"🧹 O2U: 应用已计算的噪声剔除 {drop_k}/{len(scored)} 张图片 ({O2U_DROP_FRAC*100:.1f}%)")
            train_loader, n_all, n_keep = rebuild_loader_excluding(noisy_paths, current_batch_size)
            print(f"   训练样本数: {n_all} -> {n_keep} (已剔除 {n_all - n_keep})")
            unlabeled_loader = rebuild_train_loader(BASE_BATCH_SIZE)
            o2u_applied = True
        else:
            train_loader = rebuild_train_loader(current_batch_size)
            unlabeled_loader = rebuild_train_loader(BASE_BATCH_SIZE)
        unlabeled_iter = iter(unlabeled_loader)
    else:
        print("❌ 未找到恢复文件，将从epoch 0开始训练")
        start_epoch = 0

    print(f"🎯 训练将从 epoch {start_epoch} 开始，当前batch_size: {current_batch_size}")
    print(f"📈 使用增强学习率调度: lr_min=3e-6")
    print(f"🎯 使用增强TTA: 水平翻转={TTA_HFLIP}, 多尺度={TTA_MULTI_SCALE}, 尺度={TTA_SCALES}")

    for epoch in range(start_epoch, EPOCHS):
        # ===== 动态调整batch_size =====
        if epoch == SEMI_START_EPOCH - 1 and current_batch_size != BASE_BATCH_SIZE:
            print(f"🔄 进入半监督阶段，batch_size {current_batch_size} → {BASE_BATCH_SIZE}")
            current_batch_size = BASE_BATCH_SIZE
            if o2u_scores and o2u_applied:
                scored = sorted(o2u_scores.items(), key=lambda x: x[1], reverse=True)
                drop_k = int(len(scored) * O2U_DROP_FRAC)
                noisy_paths = set([p for p, _ in scored[:drop_k]])
                train_loader, n_all, n_keep = rebuild_loader_excluding(noisy_paths, current_batch_size)
                print(f"🧹 保持O2U剔除状态，训练样本数: {n_keep}")
            else:
                train_loader = rebuild_train_loader(current_batch_size)
            unlabeled_loader = rebuild_train_loader(current_batch_size)
            unlabeled_iter = iter(unlabeled_loader)

        if epoch == FINAL_START_EPOCH - 1 and current_batch_size != FINAL_BATCH_SIZE:
            print(f"🔄 回到全监督阶段，batch_size {current_batch_size} → {FINAL_BATCH_SIZE}")
            current_batch_size = FINAL_BATCH_SIZE
            if o2u_scores and o2u_applied:
                scored = sorted(o2u_scores.items(), key=lambda x: x[1], reverse=True)
                drop_k = int(len(scored) * O2U_DROP_FRAC)
                noisy_paths = set([p for p, _ in scored[:drop_k]])
                train_loader, n_all, n_keep = rebuild_loader_excluding(noisy_paths, current_batch_size)
                print(f"🧹 保持O2U剔除状态，训练样本数: {n_keep}")
            else:
                train_loader = rebuild_train_loader(current_batch_size)
            unlabeled_loader = None
            unlabeled_iter = None

        model.train()
        in_o2u  = O2U_ENABLE and (O2U_START_EPOCH-1) <= epoch <= (O2U_END_EPOCH-1)
        use_semi = (SEMI_START_EPOCH-1) <= epoch <= (SEMI_END_EPOCH-1)
        use_final_supervised = epoch >= (FINAL_START_EPOCH-1)

        base_use_mixup = (epoch < int(EPOCHS * 0.8))
        use_mixup = base_use_mixup and (not in_o2u) and (not use_semi)

        current_lambda_u = get_lambda_u(epoch, SEMI_START_EPOCH-1, SEMI_END_EPOCH-1, LAMBDA_U_MAX)

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Mode="
              f"{'O2U' if in_o2u else ('Semi' if use_semi else ('FinalSup' if use_final_supervised else 'Sup'))} "
              f"| Mixup={'ON' if use_mixup else 'OFF'} | BatchSize={current_batch_size} "
              f"| λ_u={current_lambda_u:.3f}")

        if in_o2u:
            win_len = (O2U_END_EPOCH - O2U_START_EPOCH + 1)
            lr_cur = o2u_triangular_lr(epoch - (O2U_START_EPOCH-1), win_len, O2U_CYCLES, O2U_LR_MIN, O2U_LR_MAX)
            set_lr(optimizer, lr_cur)
            print(f"  O2U: 三角LR={lr_cur:.2e}")

        epoch_loss_sum, sample_count = 0.0, 0
        batch_cache = []

        # ========= 训练循环 =========
        progress = tqdm(
            train_loader,
            desc=f"epoch{epoch+1}",
            ncols=60,
            dynamic_ncols=False,
            leave=False,
            mininterval=0.5,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {percentage:3.0f}% "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        )

        for batch in progress:
            # ===== 统一处理训练数据 =====
            if len(batch) == 5:
                (img_g, img_2nd), labels, _, paths, mask = batch
            else:
                img_g = batch[0]
                labels = batch[1]
                paths = batch[3] if len(batch) > 3 else [f"fake_{i}" for i in range(len(img_g))]
                img_2nd = img_g
                mask = torch.zeros(len(labels), dtype=torch.uint8)

            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.uint8)
            mask = mask.bool()

            imgs_list = [img_g.to(DEVICE)]
            if mask.any():
                imgs_list.append(img_2nd[mask].to(DEVICE))
            imgs_sup = torch.cat(imgs_list, dim=0)

            labels_sup = labels.to(DEVICE)
            if mask.any():
                labels_sup = torch.cat([labels_sup, labels_sup[mask]], dim=0)

            optimizer.zero_grad(set_to_none=True)

            # ------- 监督损失 -------
            if use_mixup:
                if imgs_sup.shape[0] % 2 != 0:
                    imgs_sup   = imgs_sup[:-1]
                    labels_sup = labels_sup[:-1]
                imgs_m, targets_soft = mixup_fn(imgs_sup, labels_sup)
                with autocast_ctx():
                    logits = model(imgs_m)
                    loss_vec_sup = criterion_soft.forward(logits, targets_soft)
                    loss_sup = loss_vec_sup.mean()
            else:
                with autocast_ctx():
                    logits = model(imgs_sup)
                    loss_vec_sup = criterion_hard(logits, labels_sup)
                    loss_sup = loss_vec_sup.mean()

            loss = loss_sup

            # ------- 半监督分支 -------
            if use_semi and not use_final_supervised:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)

                if len(unlabeled_batch) == 5:
                    (u_w, u_s), _, _, _, _ = unlabeled_batch
                else:
                    u_w = unlabeled_batch[0]
                    u_s = u_w

                u_w = u_w.to(DEVICE)
                u_s = u_s.to(DEVICE)

                with torch.no_grad():
                    with autocast_ctx():
                        logits_u_w = model(u_w)
                probs_u_w = torch.softmax(logits_u_w, dim=1)
                conf, pseudo = probs_u_w.max(dim=1)

                mask_u = adaptive_threshold(conf, ADAPTIVE_RATIO, TAU)
                selected_ratio = mask_u.sum().item() / max(1, len(mask_u))

                with autocast_ctx():
                    logits_u_s = model(u_s)
                    loss_u_all = F.cross_entropy(logits_u_s, pseudo, reduction='none')
                if mask_u.sum() > 0:
                    loss_u = (loss_u_all * mask_u).sum() / mask_u.sum()
                    loss = loss + current_lambda_u * loss_u

            # ------- 反传 & EMA -------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            # ------- O2U 打分 -------
            if in_o2u:
                with torch.no_grad(), autocast_ctx():
                    logits_stat = model(imgs_sup)
                    stat_loss = criterion_hard(logits_stat, labels_sup).detach().cpu()
                epoch_loss_sum += stat_loss.sum().item()
                sample_count   += stat_loss.numel()
                batch_cache.append((paths, stat_loss[:len(paths)].clone()))

            # ========= 进度更新 =========
            if progress.n % 10 == 0:  # 每 10 步更新一次
                progress.set_postfix_str(
                    f"b={current_batch_size} λ={current_lambda_u:.3f} "
                    f"loss={loss.item():.3f} lr={optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False
                )

        # ===== 学习率调度步进 =====
        if not in_o2u:
            cosine.step(epoch + 1)

        # ------- O2U 汇总与剔除 -------
        if O2U_ENABLE and not o2u_applied:
            if in_o2u and sample_count > 0:
                epoch_mean = epoch_loss_sum / sample_count
                denom = (epoch_mean + 1e-8)
                for paths_b, loss_cpu in batch_cache:
                    norm = loss_cpu.numpy() / denom
                    for p, v in zip(paths_b, norm):
                        o2u_scores[p] = o2u_scores.get(p, 0.0) + float(v)

            if epoch == (O2U_END_EPOCH-1):
                scored = sorted(o2u_scores.items(), key=lambda x: x[1], reverse=True)
                drop_k = int(len(scored) * O2U_DROP_FRAC)
                noisy_paths = set([p for p, _ in scored[:drop_k]])
                print(f"🧹 O2U: 标记疑似噪声 {drop_k}/{len(scored)} "
                      f"({O2U_DROP_FRAC*100:.1f}%)，将从训练集中移除。")
                train_loader, n_all, n_keep = rebuild_loader_excluding(noisy_paths, current_batch_size)
                print(f"   训练样本数: {n_all} -> {n_keep} (已剔除 {n_all - n_keep})")
                unlabeled_loader = rebuild_train_loader(current_batch_size)
                unlabeled_iter = iter(unlabeled_loader)
                o2u_applied = True

        # ------- 验证：用 EMA 权重 + 增强TTA -------
        val_model = getattr(ema, 'ema', None) or getattr(ema, 'module', None) or model
        acc = evaluate(val_model, val_loader)

        if acc > best_acc:
            best_acc = acc
            torch.save(val_model.state_dict(), SAVE_PATH)
            best_ckpt_path = save_ckpt(epoch, model, ema, optimizer, scaler, cosine, best_acc, o2u_scores, current_batch_size, suffix="best")
            print(f"💾 [EMA] 保存新最优: {best_acc:.2f}% -> {SAVE_PATH} | {best_ckpt_path}")

        latest_ckpt_path = save_ckpt(epoch, model, ema, optimizer, scaler, cosine, best_acc, o2u_scores, current_batch_size, suffix="latest")
        print(f"💾 已保存最新断点：{latest_ckpt_path}")

    print("\n" + "="*50)
    print("✅ 训练完成。")
    # 最终评估 EMA 权重
    final_model = getattr(ema, 'ema', None) or getattr(ema, 'module', None) or model
    final_acc = evaluate(final_model, val_loader)
    print(f"🏁 最终 EMA 验证 Top-1: {final_acc:.2f}%")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()