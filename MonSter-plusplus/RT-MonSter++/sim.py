import onnx
import onnxsim

# 加载原始 ONNX 模型
onnx_model = onnx.load('./model.onnx')

# 使用 onnxsim 简化模型
onnx_model_simplified, check = onnxsim.simplify(onnx_model)

# 确保模型简化成功
assert check, "Simplified ONNX model could not be validated!"

# 保存简化后的模型
onnx.save(onnx_model_simplified, 'model_simplified.onnx')