import os, sys, torch, types

ROOT = os.environ.get("WORKSHOP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "TF-VPR-main"))
from model import network_infer

def make(mode, clusters, agg="tf", **kw):
    d = dict(backbone="dinov2", mode=mode, num_clusters=clusters,
             features_dim=768, work_with_tokens=False, fc_output_dim=None,
             agg_mode=agg, ot_reg=0.2, ot_iters=20, ot_temp=1.0,
             ot_dustbin=0.0, ot_dustbin_cost=0.0, ot_cost_clip=None)
    d.update(kw)
    a = types.SimpleNamespace(**d)
    m = network_infer.Infer_Model(a, pretrained_foundation=True,
                                  foundation_model_path=os.path.join(ROOT, "weights", "dinov2_vitb14_pretrain.pth"))
    return m.cuda().eval(), a

x = torch.randn(2,3,336,336, device="cuda")
for mode, c, agg, kw in [("TF_VPR",17,"tf",{}), ("TF_VPR",17,"ot",{}),
                         ("TF_VPR",17,"ot",{"ot_dustbin":0.1}),
                         ("TF_VPR",17,"ot",{"ot_cost_clip":5.0})]:
    with torch.no_grad():
        m,a = make(mode,c,agg,**kw)
        y = m(x)
    print(mode, agg, kw, "output", tuple(y.shape), "expected", a.features_dim*c,
          "norm", y.norm(dim=1).mean().item())
    del m; torch.cuda.empty_cache()
