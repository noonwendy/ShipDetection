from ultralytics import YOLO
import json

def evaluate_best_model():
    # 加载最佳模型
    model = YOLO("runs/detect/train/weights/best.pt")
    # 评估测试集（需在data.yaml中配置test路径）
    results = model.val(data="datasets/warships/data.yaml", split="test")

    # 保存评估结果到文件
    # 获取评估指标
    metrics = {
        "box": {
            "map50": results.box.map50,  # 整体 mAP50
            "map": results.box.map,  # 整体 mAP50-95
            "mp": results.box.mp,  # 整体 Precision
            "mr": results.box.mr,  # 整体 Recall
            "class_metrics": {}  # 每个类别的指标
        },
        "speed": results.speed
    }

    # 获取每个类别的指标
    for i, name in enumerate(results.names):
        p, r, ap50, ap = results.box.class_result(i)
        metrics["box"]["class_metrics"][name] = {
            "p": float(p),
            "r": float(r),
            "ap50": float(ap50),
            "ap": float(ap)
        }

    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_best_model()
