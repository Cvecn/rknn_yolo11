import warnings

warnings.filterwarnings('ignore')

import os
from pathlib import Path


def _sanitize_cuda_ld_library_path() -> None:
    """Remove user CUDA toolkit paths that can override conda PyTorch cuDNN libs."""
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    if not ld:
        return
    bad_tokens = ('/home/cx/cuda-', '/usr/local/cuda')
    cleaned = [p for p in ld.split(':') if p and not any(token in p for token in bad_tokens)]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(cleaned)


_sanitize_cuda_ld_library_path()

from ultralytics import YOLO
# -------------------------- 路径与训练配置 --------------------------
# 可改为 'yolo11n.pt' / 'yolo11s.pt' / 'yolo11m.pt' / 'yolo11l.pt' / 'yolo11x.pt'
model_init_path = 'yolo11n.pt'
# 数据集配置文件
data_yaml_path = '/home/cx/code/data/VTMOT-Highway_yolo/dataset.yaml'
# 训练输出目录
project_dir = os.path.abspath('runs/train/yolo11')
# 实验名称
run_name = 'VTMOT-Highway_yolo11_epoch100'
# 使用设备，如 '0'、'0,1' 或 'cpu'
device_num = '1,2'
# 批次大小
batch_size = 64
# 训练轮数
epochs = 100
# 断点续训权重
resume_ckpt = Path(project_dir) / run_name / 'weights' / 'last.pt'


if __name__ == '__main__':
    if resume_ckpt.exists():
        print(f'找到续训文件: {resume_ckpt}，开始续训...')
        model = YOLO(str(resume_ckpt))
        results = model.train(
            resume=True,
            imgsz=640,
            batch=batch_size,
            device=device_num,
            workers=8,
        )
    else:
        print('未找到续训文件，开始全新训练...')
        model = YOLO(model_init_path)
        results = model.train(
            data=data_yaml_path,
            imgsz=640,
            epochs=epochs,
            batch=batch_size,
            workers=8,
            device=device_num,
            project=project_dir,
            name=run_name,
        )

    metrics = model.val()
    print('训练完成，验证指标:')
    print(metrics)
