import onnx
from onnxsim import simplify

def simplify_onnx(input_path, output_path, dynamic=False):
    print("Loading ONNX:", input_path)
    model = onnx.load(input_path)

    print("Checking ONNX model...")
    onnx.checker.check_model(model)

    print("Simplifying ONNX...")
    # dynamic = True → 支持动态尺寸
    simplified_model, check = simplify(
        model,
        dynamic_input_shape=dynamic
    )

    if not check:
        raise RuntimeError("Simplified ONNX model could NOT be validated!")

    print("Saving simplified model:", output_path)
    onnx.save(simplified_model, output_path)

    print("Simplification Done!")

    # 打印节点数量对比
    print(f"Before: {len(model.graph.node)} nodes")
    print(f"After : {len(simplified_model.graph.node)} nodes")


if __name__ == "__main__":
    input_onnx = "onnx_save/S2M2_S_1280_736_v2_torch29.onnx"
    output_onnx = "onnx_save/S2M2_S_1280_736_v2_torch29_sim.onnx"

    # 若模型是固定输入尺寸（你的 stereo 模型一般是固定输入）
    simplify_onnx(input_onnx, output_onnx, dynamic=False)

    # 如果你是动态输入模型，则使用 dynamic=True
    # simplify_onnx(input_onnx, output_onnx, dynamic=True)
