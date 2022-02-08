## YOLOv4 Tiny Pytorch & ONNX & TensorRT
---


## Tutorial
https://github.com/bubbliiiing/yolov4-tiny-pytorch/blob/master/README.md


## Convert .pth file to ONNX
```
python convert_onnx.py --model_name [...]
```

## How to run with TensorRT
### Environment
You can run in docker or in your local enviroment. Recomended run with docker.    
Install NVIDIA docker for Tensorrt
```
docker pull nvcr.io/nvidia/tensorrt:21.07-py3 
```
Run container
```
docker run -it --runtime=nvidia -v $(pwd):/yolo-tiny --name yolov4-tiny nvcr.io/nvidia/tensorrt:21.07-py3
docker exec nvcr -it yolov4-tiny bash
```
### Build TRT serialized file
```
python3 build_engine.py -m yolo-tiny
```
### Inference
```
python3 infer_trt.py
```

## Reference
https://github.com/bubbliiiing/yolov4-tiny-pytorch  
  
