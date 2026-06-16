
# 480 x 1280 
## fp16
trtexec --onnx=./igev_480_1280.onnx --saveEngine=./igev_480_1280_fp16.engine --fp16
## int8
trtexec --onnx=./igev_480_1280.onnx --saveEngine=./igev_480_1280_int8.engine --fp16 --int8

# 736 x 1280
trtexec --onnx=./igev_720_1280.onnx --saveEngine=./igev_720_1280_int8.engine --fp16 --int8
