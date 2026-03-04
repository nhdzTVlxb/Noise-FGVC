import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from tqdm import tqdm
import warnings
from model import get_model

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE_VAL = 320

def five_crop_collate_fn(batch):
    """自定义 collate_fn 处理 FiveCrop 数据"""
    imgs5_list = []
    names_list = []
    
    for item in batch:
        imgs5, name = item
        imgs5_list.append(imgs5)  # [5, C, H, W]
        names_list.append(name)
    
    # 手动拼接 tensor
    batch_imgs5 = torch.stack(imgs5_list, dim=0)  # [batch_size, 5, C, H, W]
    return batch_imgs5, names_list

class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.img_paths = [
            os.path.join(test_dir, fname)
            for fname in sorted(os.listdir(test_dir))
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]
        print(f"📁 找到 {len(self.img_paths)} 张测试图像")

    def _to_tensor_and_norm(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"⚠️ 加载图像失败 {path}: {e}")
            img = Image.new("RGB", (IMG_SIZE_VAL, IMG_SIZE_VAL), (128, 128, 128))

        # 使用固定尺度 320（简化处理，避免多尺度复杂度）
        scale = 320
        resize_side = 360  # 320 * 1.125 = 360
        
        # FiveCrop 变换
        val_tfms = transforms.Compose([
            transforms.Resize((resize_side, resize_side),
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.FiveCrop(scale),
            transforms.Lambda(lambda crops: torch.stack(
                [self._to_tensor_and_norm()(c) for c in crops], dim=0))
        ])
        
        imgs5 = val_tfms(img)  # [5, C, H, W] = [5, 3, 320, 320]
        return imgs5, os.path.basename(path)

    def __len__(self):
        return len(self.img_paths)

def inference(model_path, test_dir, output_path, num_classes=5000, batch_size=80):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🧠 加载模型：{model_path}")
    print(f"📂 测试集路径：{test_dir}")
    print(f"💾 输出文件：{output_path}")
    print(f"⚙️ 运行设备：{device}")

    # 加载模型
    model = get_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device).eval()

    # 数据加载 - 使用自定义 collate_fn
    test_dataset = TestDataset(test_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        collate_fn=five_crop_collate_fn  # 关键：使用自定义collate函数
    )

    # 推理
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        #f.write("image_name,predicted_label\n")
        with torch.no_grad():
            for imgs5, names in tqdm(test_loader, desc="Predicting"):
                # imgs5 shape: [batch_size, 5, 3, 320, 320]
                batch_size, num_crops, C, H, W = imgs5.shape
                
                # 重塑为 [batch_size * 5, 3, 320, 320]
                imgs_flat = imgs5.view(batch_size * num_crops, C, H, W).to(device)
                
                # 前向传播
                outputs = model(imgs_flat)  # [batch_size * 5, num_classes]
                
                # 重塑回 [batch_size, 5, num_classes]
                outputs = outputs.view(batch_size, num_crops, -1)
                
                # 对5个crop的预测结果取平均
                avg_outputs = outputs.mean(dim=1)  # [batch_size, num_classes]
                
                # 取最大概率的类别
                preds = torch.argmax(avg_outputs, dim=1).cpu().numpy()
                
                for n, p in zip(names, preds):
                    f.write(f"{n},{p:04d}\n")
    
    print(f"\n✅ 推理完成！结果已保存至：{output_path}")
    print(f"📊 处理了 {len(test_dataset)} 张图像")

if __name__ == "__main__":
    model_path = "/home/node/zzz/checks/model5000o2u.pth"
    test_dir = "/home/node/webinat5000_test_B/test_B"
    output_path = "/home/node/zzz/pred_results_web5000.csv"
    num_classes = 5000

    inference(model_path, test_dir, output_path, num_classes=num_classes, batch_size=80)