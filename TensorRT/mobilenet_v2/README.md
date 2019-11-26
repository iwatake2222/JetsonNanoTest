# TensorRT (C++) sample code for MobileNet v2 (image classification) on Jetson Nano

## Environment
- Jetson Nano
	- Jetpack 4.2.1 [L4T 32.2.0]
	- No need to install any other software
- Raspberry Pi Camera v2
- The model is retrieved from the following:
	- https://github.com/onnx/models/tree/master/vision/classification/mobilenet
		- https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx
	- Model details
		- MobileNet v2
		- 224x224x3
		- 1000 classes

## Performance
- Inference time
	- FP32: 19.53 [msec]
	- FP16: 17.65 [msec]
- conditions
	- `sudo nvpmodel -m 0`
	- `sudo jetson_clocks`

## FLAGS
- `CONVERT_ONNX_TO_TRT`
	- use onnx model, and save trt model
- `CONVERT_UFF_TO_TRT`
	- use uff model, and save trt model
	- not supported yet
- None of the aboves
	- use trt model
