# from ultralytics.yolo.utils.benchmarks import benchmark
from ultralytics.yolo.utils.plotting import plot_results

# Benchmark on GPU
# benchmark(model='D:/IntelligentSystem/YOLOV8/runs/detect/train16/weights/best.pt', imgsz=640, half=False, device=0)


plot_results(file="C:/Users/Asus/Downloads/train16-20230430T213842Z-001/train30/results.csv")