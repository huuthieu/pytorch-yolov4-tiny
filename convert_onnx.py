import os
import json
import argparse

from yolo import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert pth to onnx')
    parser.add_argument('--model_name', type = str, default = "yolo-tiny.onnx")
    args = parser.parse_args()
    with open('config/config.json','r') as f:
        cfg = json.load(f)
        weight_folder = os.path.join(cfg['logs'], "yolov4-tiny-best-weights.pth")

    yolo = YOLO(model_path = weight_folder, confidence = cfg['thresh'],
                    anchors_mask = cfg['anchors_mask'], phi = cfg['phi'],
                    input_shape = cfg['inference_input_shape'], nms_iou = cfg['nms_iou'],
                    letterbox_image = cfg['letterbox_image'])

    yolo.export_onnx(model_name = args.model_name)