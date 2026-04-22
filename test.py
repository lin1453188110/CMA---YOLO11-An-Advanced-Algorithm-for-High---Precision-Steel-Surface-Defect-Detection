import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp75/weights/best.pt') # select your model.pt path
    model.track(source=r'E:\yolo\yolov11-main\datasets\net15\images\val2017',
                imgsz=640,
                project='runs/track',
                name='exp',
                save=True
                )