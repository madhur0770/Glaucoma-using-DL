"""Weight-only fp16 storage (Cast to fp32 inside the graph): half the size, practically lossless."""
import onnx, numpy as np, sys
from onnx import numpy_helper, helper, TensorProto
def h(src, dst):
    m = onnx.load(src); g = m.graph; new = []; keep = []
    for i in g.initializer:
        W = numpy_helper.to_array(i)
        if W.dtype == np.float32 and W.size >= 1024:
            g_name = i.name + "_h"; keep.append(numpy_helper.from_array(W.astype(np.float16), g_name))
            new.append(helper.make_node("Cast", [g_name], [i.name], to=TensorProto.FLOAT, name=i.name + "_cast"))
        else: keep.append(i)
    del g.initializer[:]; g.initializer.extend(keep); nodes = list(g.node); del g.node[:]; g.node.extend(new + nodes)
    onnx.checker.check_model(m); onnx.save(m, dst)
for n in sys.argv[1:]: h(f"models/{n}.fp32.onnx", f"models/{n}.onnx")
