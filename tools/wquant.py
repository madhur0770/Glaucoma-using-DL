"""Weight-only int8 compression: every Conv/Gemm weight is stored as int8 + per-output-channel fp32 scale and
de-quantised inside the graph (DequantizeLinear). Compute stays fp32, so accuracy is essentially unchanged; files ~4x smaller."""
import onnx, numpy as np, sys
from onnx import numpy_helper, helper
def q(src, dst):
    m = onnx.load(src); g = m.graph; inits = {i.name: i for i in g.initializer}
    users = {}
    for n in g.node:
        if n.op_type in ("Conv", "Gemm", "ConvTranspose") and len(n.input) > 1 and n.input[1] in inits: users.setdefault(n.input[1], []).append(n)
    new_nodes = []; removed = set()
    for name, ns in users.items():
        W = numpy_helper.to_array(inits[name]).astype(np.float32)
        if W.size < 1024: continue
        flat = W.reshape(W.shape[0], -1); s = np.abs(flat).max(1) / 127.0; s[s == 0] = 1e-8
        Q = np.clip(np.round(flat / s[:, None]), -127, 127).astype(np.int8).reshape(W.shape)
        qn, sn, zn = name + "_q", name + "_s", name + "_z"
        g.initializer.extend([numpy_helper.from_array(Q, qn), numpy_helper.from_array(s.astype(np.float32), sn), numpy_helper.from_array(np.zeros(W.shape[0], np.int8), zn)])
        new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn, zn], [name + "_dq"], axis=0, name=name + "_dqnode"))
        for n in g.node:                      # rewire every consumer (Conv, Identity, ...) to the de-quantised tensor
            for k, inp in enumerate(n.input):
                if inp == name: n.input[k] = name + "_dq"
        removed.add(name)
    keep = [i for i in g.initializer if i.name not in removed]; del g.initializer[:]; g.initializer.extend(keep)
    nodes = list(g.node); del g.node[:]; g.node.extend(new_nodes + nodes)
    onnx.checker.check_model(m); onnx.save(m, dst)
for n in ["localiser", "discup", "tight", "ppa", "cnn", "ring"]:
    q(f"models/{n}.fp32.onnx", f"models/{n}.onnx")
import os; [print(f, round(os.path.getsize("models/"+f)/1e6,1)) for f in sorted(os.listdir("models")) if f.endswith(".onnx") and "fp32" not in f]
