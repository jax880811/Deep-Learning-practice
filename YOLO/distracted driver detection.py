from ultralytics import YOLO
import os

# 設定資料集路徑
train_images_path = r"E:\vscode\deep learning\YOLO\Distracted Driver Detection\train\images"
train_labels_path = r"E:\vscode\deep learning\YOLO\Distracted Driver Detection\train\labels"
valid_images_path = r"E:\vscode\deep learning\YOLO\Distracted Driver Detection\valid\images"
valid_labels_path = r"E:\vscode\deep learning\YOLO\Distracted Driver Detection\valid\labels"
data_yaml_path = r"E:\vscode\deep learning\YOLO\Distracted Driver Detection\data.yaml"



# 載入YOLOv8預訓練的 YOLOv8 模型
model = YOLO("yolov8n.pt")

# 訓練模型
results = model.train(
    data=data_yaml_path,  # data.yaml 文件路徑
    epochs=5,            # 訓練數 抓5次就好 沒辦法，畢竟是用cpu再跑
    batch=16,             # 一次抓16張進去
    imgsz=640,            # 圖片大小，原本圖片是640x480
    device="cpu",         # 沒辦法啊，我真的就抓不到我自己的gpu
    name="yolov8_distracted_driver_detection",  # 訓練結果的保存名稱
    save=True,            # 保存訓練結果
    exist_ok=True         # 如果結果文件夾已存在，允許覆蓋
)

# 保存訓練好的模型
model_path = r"E:\vscode\deep learning\YOLO\Distracted_Driver_Detection\yolov8_distracted_driver_detection_model.pt"
model.export(format="pt")  # 導出模型為PyTorch格式
print("模型保存位置: " + model_path)

# 評估模型效能
metrics = model.val()  # 使用驗證集進行評估
print("模型評估結果:")
print("mAP50-95: " + metrics.box.map)  # mAP50-95
print("mAP50: " + metrics.box.map50)   # mAP50
print("精確度: " + metrics.box.precision)  
print("召回率: "+ metrics.box.recall) 