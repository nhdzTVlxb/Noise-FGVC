import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import RandAugment
from PIL import Image, ImageFile
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def set_seed(seed=42, reproducible=True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark = not reproducible

# -------- 可调参数 --------
USE_STRATIFY = True
USE_WEIGHTED_SAMPLER = "auto"
IMBALANCE_RATIO_THRESHOLD = 10.0

IMG_SIZE_TRAIN = 320
IMG_SIZE_RESIZE_TRAIN = 384
IMG_SIZE_VAL = 320

GLOBAL_SCALE = (0.7, 1.0)
LOCAL_SCALE  = (0.25, 0.6)
MICRO_SCALE  = (0.08, 0.20)   # 新增：更微的局部视角

DUAL_VIEW_RATIO = 0.30        # 仅 30% 样本使用双视角

VAL_TTA_SCALES = [288, 320, 352]
VAL_TTA_PROBS  = [0.2, 0.6, 0.2]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class WebFineGrainedDataset(Dataset):
    """
    train:
        return ((img_global, img_2nd), label, weight, path, two_view_mask)
        - two_view_mask=1: 第2视角为局部/微局部
        - two_view_mask=0: 第2视角仅占位（等于全局），训练时不要搬上 GPU
    val/test:
        return (five_crops[5,C,H,W], label, weight, path)
    """
    def __init__(self, data_root, split="train"):
        self.data_root = data_root
        self.split = split
        self.classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.img_paths, self.labels = self._gather()
        self._build_tfms()

    def _gather(self):
        paths, labels = [], []
        for c in self.classes:
            d = os.path.join(self.data_root, c)
            for n in sorted(os.listdir(d)):
                if n.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                    paths.append(os.path.join(d, n))
                    labels.append(self.class_to_idx[c])

        if self.split in ["train", "val"]:
            if USE_STRATIFY:
                tr_p, va_p, tr_l, va_l = train_test_split(
                    paths, labels, test_size=0.1, stratify=labels, random_state=42
                )
            else:
                tr_p, va_p, tr_l, va_l = train_test_split(
                    paths, labels, test_size=0.1, stratify=None, random_state=42
                )
            return (tr_p, tr_l) if self.split == "train" else (va_p, va_l)

        return paths, [0] * len(paths)

    def _norm(self): return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    def _to_tensor_and_norm(self): return transforms.Compose([transforms.ToTensor(), self._norm()])

    def _build_tfms(self):
        if self.split == "train":
            base = [
                transforms.Resize((IMG_SIZE_RESIZE_TRAIN, IMG_SIZE_RESIZE_TRAIN),
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                RandAugment(num_ops=2, magnitude=9),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomGrayscale(0.08),
                
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            ]
            self.global_tfms = transforms.Compose([
                *base,
                transforms.RandomResizedCrop(IMG_SIZE_TRAIN, scale=GLOBAL_SCALE,
                                             interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(), self._norm(),
            ])
            self.local_tfms = transforms.Compose([
                *base,
                transforms.RandomResizedCrop(IMG_SIZE_TRAIN, scale=LOCAL_SCALE,
                                             interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(), self._norm(),
            ])
            self.micro_tfms = transforms.Compose([
                *base,
                transforms.RandomResizedCrop(IMG_SIZE_TRAIN, scale=MICRO_SCALE,
                                             interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(), self._norm(),
            ])
            # 多次小面积擦除（Hide-and-Seek 风格）
            self.erase_many = [transforms.RandomErasing(p=0.15, scale=(0.02, 0.10),
                                 ratio=(0.3, 3.3), value='random') for _ in range(3)]
        else:
            pass  # val/test 在 __getitem__ 动态构造 TTA

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        p, y = self.img_paths[idx], self.labels[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE_TRAIN, IMG_SIZE_TRAIN), (255, 255, 255))

        if self.split == "train":
            img_g = self.global_tfms(img)
            use_dual = (random.random() < DUAL_VIEW_RATIO)
            if use_dual:
                img_2nd = self.local_tfms(img) if random.random() < 0.5 else self.micro_tfms(img)
                for er in self.erase_many:
                    img_g = er(img_g); img_2nd = er(img_2nd)
                mask = torch.tensor(1, dtype=torch.uint8)
            else:
                img_2nd = img_g
                for er in self.erase_many: img_g = er(img_g)
                mask = torch.tensor(0, dtype=torch.uint8)
            return (img_g, img_2nd), torch.tensor(y, dtype=torch.long), 1.0, p, mask

        # val/test：三尺度随机 + FiveCrop，主尺度 320 概率更高
        scale = 320
        resize_side = int(round(scale * 1.125))  # 与 320->360 保持一致
        val_tfms = transforms.Compose([
            transforms.Resize((resize_side, resize_side),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.FiveCrop(scale),
            transforms.Lambda(lambda crops: torch.stack(
                [self._to_tensor_and_norm()(c) for c in crops], dim=0))
        ])
        imgs5 = val_tfms(img)  # [5,C,H,W]，此处为 [5,3,320,320]
        return imgs5, torch.tensor(y, dtype=torch.long), 1.0, p

def _need_weighted_sampler(labels):
    arr = np.array(labels, dtype=np.int64)
    counts = np.bincount(arr); counts = counts[counts > 0]
    if len(counts) == 0: return False, None, None
    max_c, min_c = int(counts.max()), int(counts.min())
    ratio = float(max_c) / float(min_c) if min_c > 0 else float('inf')
    return (ratio >= IMBALANCE_RATIO_THRESHOLD), ratio, counts

def get_dataloader(data_root, split="train", batch_size=32, num_workers=8,
                   use_weighted_sampler=USE_WEIGHTED_SAMPLER):
    ds = WebFineGrainedDataset(data_root, split=split)
    sampler = None; shuffle = (split == "train")

    if split == "train":
        enable_sampler = False
        if use_weighted_sampler == "auto":
            auto_flag, ratio, _ = _need_weighted_sampler(ds.labels)
            enable_sampler = auto_flag
            if auto_flag:
                print(f"[Sampler] 检测到不均衡（max/min≈{ratio:.2f}），启用 WeightedRandomSampler。")
        elif use_weighted_sampler is True:
            enable_sampler = True

        if enable_sampler:
            class_counts = np.bincount(ds.labels); class_counts[class_counts == 0] = 1
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[ds.labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.float32),
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=(split=="train"),
        prefetch_factor=2, persistent_workers=True, collate_fn=None
    )
    return loader, ds.class_to_idx
