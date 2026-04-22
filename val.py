import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp7/weights/best.pt')
    model.val(data=r"E:\yolo\yolov11-main\datasets\net15\my21.yaml",
              split='val',
              imgsz=640,
              batch=4,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )