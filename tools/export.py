import sys, torch, timm, segmentation_models_pytorch as smp, numpy as np, onnx, json, os
"""Export the trained PyTorch checkpoints (models/*.pt from the training scripts) to ONNX for the browser demo.
Run from the repo root: python tools/export.py [names], then compress with tools/hquant.py / tools/wquant.py."""
M = "models/"; O = "docs/models/"
def ld(m, f): m.load_state_dict(torch.load(M + f, map_location="cpu")); return m.eval()
class Seg(torch.nn.Module):          # sigmoid + flip-TTA inside the graph, so the browser gets probabilities directly
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return (torch.sigmoid(s.m(x)) + torch.sigmoid(s.m(torch.flip(x, [3]))).flip(3)) / 2
class Cls(torch.nn.Module):          # returns logit (TTA mean prob) and layer-4 features for CAM
    def __init__(s, m, cam=False): super().__init__(); s.m = m; s.cam = cam
    def forward(s, x):
        f = s.m.forward_features(x); lg = s.m.forward_head(f)
        if s.cam:      # mean logit over the image and its mirror (as used to fit the decision model) + features for CAM
            return (lg + s.m.forward_head(s.m.forward_features(torch.flip(x, [3])))) / 2, f
        f2 = s.m.forward_features(torch.flip(x, [3])); lg2 = s.m.forward_head(f2)
        return (torch.sigmoid(lg) + torch.sigmoid(lg2)) / 2
nets = {
 "localiser": (Seg(ld(smp.Unet("resnet34", encoder_weights=None, classes=1), "seg_full512_base.pt")), 512),
 "discup": (Seg(ld(smp.Unet("resnet34", encoder_weights=None, classes=2), "seg_crop512_base.pt")), 512),
 "tight": (Seg(ld(smp.UnetPlusPlus("resnet34", encoder_weights=None, classes=2), "seg_tight_unetpp.pt")), 512),
 "ppa": (Seg(ld(smp.Unet("resnet34", encoder_weights=None, classes=2), "ppa_supervised.pt")), 512),
 "cnn": (Cls(ld(timm.create_model("resnet50", pretrained=False, num_classes=1), "cls_resnet50.pt"), cam=True), 384),
 "ring": (Cls(ld(timm.create_model("resnet18", pretrained=False, num_classes=1), "cls_ring_resnet18.pt")), 384),
}
for n, (net, s) in [(k, v) for k, v in nets.items() if k in (sys.argv[1:] or nets)]:
    x = torch.randn(1, 3, s, s)
    outs = ["prob"] if n not in ("cnn",) else ["logit", "features"]
    torch.onnx.export(net, x, O + n + ".fp32.onnx", input_names=["input"], output_names=outs, opset_version=17, dynamo=False)
    print(n, os.path.getsize(O + n + ".fp32.onnx") / 1e6, "MB")
fc = nets["cnn"][0].m.get_classifier(); json.dump({"w": fc.weight.detach().numpy()[0].tolist(), "b": float(fc.bias.detach()[0])}, open(O + "cnn_fc.json", "w"))
