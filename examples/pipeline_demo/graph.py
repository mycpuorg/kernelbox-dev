import torch
import operator

def run(input):
    out = {}
    out["gelu_1"] = torch.ops.aten.gelu.default(input["x"])
    out["add_1"] = torch.ops.aten.add.Tensor(out["gelu_1"], input["bias"])
    out["mm_1"] = torch.ops.aten.mm.default(out["add_1"], input["weight"])
    out["relu_1"] = torch.ops.aten.relu.default(out["mm_1"])
    out["view_1"] = torch.ops.aten.view.default(out["relu_1"], [2, 16, 32])
    out["mul_1"] = torch.ops.aten.mul.Tensor(out["view_1"], input["scale"])
    return [out["mul_1"]]
