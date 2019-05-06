Test code for Jetson Nano


# Speed Test
## Inference of Mobilenet(224x224, retrieved using mmdownload)
- Python, Tensorflow.keras (CPU): 162.3msec
- Python, Tensorflow.keras (GPU): 61.8msec
- C++, ncnn (CPU): 48.1msec
- C++, ncnn (GPU): 44.8msec


