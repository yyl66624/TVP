import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel

from model.cct import cct_14_7x2_384
from model.normalization import L2Norm
import model.free_module as free
from model.ot_cam import OT_CAM, SoftmaxTokenCAM

from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
import torch.nn.functional as F
import math
from model.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

from torch.nn.parameter import Parameter
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class Infer_Model(nn.Module):
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        super().__init__()
        self.backbone = get_backbone(args, pretrained_foundation, foundation_model_path)
        self.arch_name = args.backbone
        self.mode = args.mode
        self.resize = tuple(getattr(args, "resize", [336, 336]))
        if args.fc_output_dim != None:
            args.features_dim = args.fc_output_dim

        # VLAD mode (TFA-style base descriptor): [CLS, GeM, mean, K VLAD blocks].
        self.vlad_k = int(getattr(args, "vlad_k", 64))
        self.vlad_soft = float(getattr(args, "vlad_soft", 0.0) or 0.0)
        if getattr(args, "vlad_centers", ""):
            _c = np.load(args.vlad_centers).astype("float32")
            self.vlad_k = int(_c.shape[0])
            self.register_buffer("vlad_centers_t", torch.from_numpy(_c))
        # Optional second codebook (TF_VLAD mode only): two VLAD blocks with
        # different K / seed are complementary after calibration.
        self.vlad_k2 = 0
        if getattr(args, "vlad_centers2", ""):
            _c2 = np.load(args.vlad_centers2).astype("float32")
            self.vlad_k2 = int(_c2.shape[0])
            self.register_buffer("vlad_centers2_t", torch.from_numpy(_c2))

        self.agg_GeM = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())
        # TF-VPR outputs (num_clusters - 1) cluster tokens + 1 CLS token, so the
        # total descriptor dimension is num_clusters * features_dim (see Table 2,
        # "17 x 768").  Other modes use num_clusters as the number of output
        # tokens directly.
        self.tf_variant = getattr(args, "tf_variant", "n_minus_1_cls")
        self.tf_global_token = getattr(args, "tf_global_token", "cls")
        self.add_cls = self.tf_global_token != "none"
        if args.mode in ("TF_VPR", "TF_VLAD"):
            if self.tf_variant == "n_minus_1_cls":
                clusters_num = args.num_clusters - 1
            elif self.tf_variant == "n_cls":
                clusters_num = args.num_clusters
            else:  # n_no_cls
                clusters_num = args.num_clusters
                self.add_cls = False
                self.tf_global_token = "none"
        else:
            clusters_num = args.num_clusters
        if args.mode in ("GeM_CLS", "GeM_Mean"):
            self.out_tokens = 14
        elif args.mode == "CLS_Mean":
            self.out_tokens = 2
        elif args.mode in ("CLS", "Mean"):
            self.out_tokens = 1
        else:
            extra = 2 if self.tf_global_token == "both" else (1 if self.add_cls else 0)
            self.fuse_gem = bool(getattr(args, "tf_fusion_gem", False))
            self.mean_view = bool(getattr(args, "tf_mean_view", False))
            self.lang_tokens = (int(getattr(args, "lang_tokens", 0))
                                if getattr(args, "lang_prompt_dir", None) else 0)
            self.out_tokens = (clusters_num + extra + (13 if self.fuse_gem else 0)
                               + (1 if self.mean_view else 0) + self.lang_tokens)
        if args.mode == "VLAD":
            self.out_tokens = 3 + self.vlad_k
        if args.mode == "TF_VLAD":
            # TF cluster tokens (+CLS) and multi-scale GeM cells, then the VLAD
            # blocks -- one dump, three complementary view families
            self.out_tokens = self.out_tokens + self.vlad_k + self.vlad_k2
        agg_mode = getattr(args, "agg_mode", "tf")
        if agg_mode == "st":
            self.agg_cross = SoftmaxTokenCAM(
                clusters_num=clusters_num,
                dim=args.features_dim,
                work_with_tokens=args.work_with_tokens,
                temperature=getattr(args, "st_temp", 1.0),
            )
        elif agg_mode == "ot":
            self.agg_cross = OT_CAM(
                clusters_num=clusters_num,
                dim=args.features_dim,
                work_with_tokens=args.work_with_tokens,
                ot_reg=args.ot_reg,
                ot_iters=args.ot_iters,
                ot_temp=args.ot_temp,
                dustbin=args.ot_dustbin,
                dustbin_cost=args.ot_dustbin_cost,
                cost_clip=args.ot_cost_clip,
            )
        else:
            self.agg_cross = free.TF_CAM(
                                        clusters_num=clusters_num,
                                        dim=args.features_dim,
                                        work_with_tokens=args.work_with_tokens)
        self.agg_cross.norm_first = getattr(args, "cam_norm_first", False)
        self.agg_cross.temperature = getattr(args, "tf_temp", 1.0)

        self.agg_Graph = free.TF_GAM(temperature=1)
        self.gam_k = int(getattr(args, "gam_k", 8))
        self.gam_lambda = float(getattr(args, "gam_lambda", 0.8))
        self.skip_gam = bool(getattr(args, "skip_gam", False))
        self.lang_bank = None
        if getattr(args, "lang_prompt_dir", None):
            from model.lang_prompt import LanguagePromptBank
            self.lang_bank = LanguagePromptBank(args.lang_prompt_dir)
            self.lang_gamma = args.lang_gamma
            self.lang_temp = float(getattr(args, "lang_temp", 0.1))
            ids = [0, 1, 3, 6, 9, 11, 14, 16, 17, 19, 21, 24]
            self.lang_ids = ids[: self.lang_tokens]

    def _gam(self, tokens):
        if self.skip_gam:
            return torch.nn.functional.normalize(tokens, p=2, dim=-1)
        return self.agg_Graph(tokens, node=self.gam_k, Lambda=self.gam_lambda)

    def _vlad(self, tokens, centers=None):
        """VLAD over (optionally spherical) patch tokens.

        Hard assignment (default) or soft assignment with temperature
        ``vlad_soft`` > 0.  Intra-normalise each residual block, then return the
        K x D blocks, as in Jegou et al. and the TFA reference.
        """
        B, P, D = tokens.shape
        if centers is None:
            centers, k = self.vlad_centers_t, self.vlad_k
        else:
            k = centers.shape[0]
        t = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        c = torch.nn.functional.normalize(centers, p=2, dim=-1)
        c = c.to(device=t.device, dtype=t.dtype)
        logits = t @ c.t()                                   # B, P, K
        if self.vlad_soft > 0:
            a = torch.softmax(self.vlad_soft * logits, dim=-1)
        else:
            assign = logits.argmax(dim=-1)
            a = torch.nn.functional.one_hot(assign, k).to(t.dtype)
        counts = a.sum(dim=1)                                # B, K
        sums = torch.einsum("bpk,bpd->bkd", a, t)            # B, K, D
        v = sums - counts.unsqueeze(-1) * c.unsqueeze(0)
        v = torch.nn.functional.normalize(v, p=2, dim=-1)    # intra-normalise
        return v

    @staticmethod
    def _grid_cell(x, n, i, j):
        """Crop cell (i, j) of an n x n proportional partition of a feature map."""
        H, W = x.shape[2], x.shape[3]
        r0, r1 = round(H * i / n), round(H * (i + 1) / n)
        c0, c1 = round(W * j / n), round(W * (j + 1) / n)
        return x[:, :, r0:r1, c0:c1]

    def _patch_grid(self, num_patches):
        """Spatial shape of the patch grid (W, H).

        DINOv2 accepts non-square inputs, so the square assumption
        (W = H = sqrt(P)) only holds for square resizes.
        """
        ph, pw = self.resize[0] // 14, self.resize[1] // 14
        if ph * pw == num_patches:
            return pw, ph
        s = int(math.sqrt(num_patches))
        return s, s

    def forward(self, x):
        images = x  # keep the (ImageNet-normalised) input for the prompt bank
        if self.arch_name.startswith("cct"):
            x = self.backbone(x,True)
        else:
            x = self.backbone(x)
        if self.arch_name.startswith("vit"):
            B,P,D = x.last_hidden_state.shape
            W = H = int(math.sqrt(P-1))
            x1 = x.last_hidden_state[:, 1:, :]
            x0 = x.last_hidden_state[:, 0, :]  # shape: (B, D)

            if self.mode == "TF_VPR":
                x1 = self.agg_Graph(x1)      # TF_GAM --> Training-Free Graph-Attention Module
                x1 = x1.view(B,W,H,D).permute(0, 3, 1, 2)
                x = self.agg_cross(x1)    # TF_CAM  --> Training-Free Cross-Attention Network
                x = torch.cat([x0,x], dim=-1) 
            elif self.mode == "CLS":
                x = x0

        elif self.arch_name.startswith("cct384"):
            x1, x0 = x
            B,P,D = x1.shape

            if self.mode == "TF_VPR":
                W = H = int(math.sqrt(P))
                x1 = self.agg_Graph(x1)       # TF_GAM --> Training-Free Graph-Attention Module
                x1 = x1.view(B,W,H,D).permute(0, 3, 1, 2)
                x = self.agg_cross(x1)      # TF_CAM  --> Training-Free Cross-Attention Network
                x = torch.cat([x0,x], dim=-1)
            elif self.mode == "CLS":
                x = x0

        elif self.arch_name.startswith("dinov2"):
            B,P,D = x["x_prenorm"].shape
            W, H = self._patch_grid(P - 1)
            x0 = x["x_norm_clstoken"]
            tokens = x["x_norm_patchtokens"]
            x1 = tokens.reshape(B,W,H,D).permute(0, 3, 1, 2).contiguous()

            # GeM descriptors (only needed by the non TF_VPR modes)
            if self.mode in ("GeM_CLS", "GeM_Mean", "CLS_Mean", "Mean"):
                x2 = self.agg_GeM(x1)

            if self.mode == "VLAD":
                cls_tok = torch.nn.functional.normalize(x0, p=2, dim=-1)
                gem_tok = self.agg_GeM(x1)
                mean_tok = torch.nn.functional.normalize(tokens.mean(dim=1), p=2, dim=-1)
                vlad = self._vlad(tokens)                    # B, K, D
                x = torch.cat([cls_tok.unsqueeze(1), gem_tok.unsqueeze(1),
                               mean_tok.unsqueeze(1), vlad], dim=1)

            if self.mode in ("TF_VPR", "TF_VLAD"):
                tokens_raw = tokens
                x1_raw = x1  # raw patch grid, for the multi-scale GeM tokens
                # Natural-language prompt salience (optional, training-free)
                if self.lang_bank is not None:
                    gate = self.lang_bank.token_gate(images, W, self.lang_gamma)
                    tokens = tokens * gate
                # TF_GAM --> Training-Free Graph-Attention Module
                tokens = self._gam(tokens)
                x1 = tokens.reshape(B,W,H,D).permute(0, 3, 1, 2).contiguous()
                # TF_CAM  --> Training-Free Cross-Attention Network
                x = self.agg_cross(x1).reshape(B, -1, D)
                # global token(s) + TF cluster tokens
                glob = []
                if self.tf_global_token == "cls":
                    glob = [x0.unsqueeze(1)]
                elif self.tf_global_token in ("gem", "avg", "both"):
                    gem_tok = self.agg_GeM(tokens.reshape(B,W,H,D).permute(0,3,1,2).contiguous())
                    if self.tf_global_token == "gem":
                        glob = [gem_tok.unsqueeze(1)]
                    elif self.tf_global_token == "avg":
                        mixed = F.normalize(x0 + gem_tok, p=2, dim=-1)
                        glob = [mixed.unsqueeze(1)]
                    else:
                        glob = [x0.unsqueeze(1), gem_tok.unsqueeze(1)]
                x = torch.cat(glob + [x], dim=1)
                if self.lang_bank is not None and self.lang_tokens > 0:
                    sem = self.lang_bank.semantic_tokens(images, tokens, self.lang_ids, self.lang_temp)
                    x = torch.cat([x, sem], dim=1)
                if self.fuse_gem:
                    cells = [self.agg_GeM(self._grid_cell(x1_raw, n, i, j)).unsqueeze(1)
                             for n in (2, 3) for i in range(n) for j in range(n)]
                    x = torch.cat([x] + cells, dim=1)
                if self.mean_view:
                    # global (GeM-p) pooled view of the raw patch tokens
                    x = torch.cat([x, self.agg_GeM(x1_raw).unsqueeze(1)], dim=1)
                if self.mode == "TF_VLAD":
                    # VLAD blocks from the raw patch tokens (not GAM-refined):
                    # the two families should stay as decorrelated as possible
                    blocks = [self._vlad(tokens_raw)]
                    if self.vlad_k2 > 0:
                        blocks.append(self._vlad(tokens_raw, self.vlad_centers2_t))
                    x = torch.cat([x] + blocks, dim=1)

            # choose 2/3: GeM + CLS / Mean token.  The 2x2 and 3x3 grids are
            # split proportionally so the descriptor is defined for any patch
            # resolution (the upstream code hard-coded a 24x24 grid).
            if self.mode in ("GeM_CLS", "GeM_Mean"):
                cells = []
                for n in (2, 3):
                    for i in range(n):
                        for j in range(n):
                            cells.append(self.agg_GeM(self._grid_cell(x1, n, i, j)))
                first = x0 if self.mode == "GeM_CLS" else x2
                x = torch.cat([t.unsqueeze(1) for t in [first] + cells], dim=1)

            if self.mode == "CLS_Mean":
                # choose 4:  cls_token + Mean token
                x = torch.cat([x0.unsqueeze(1),x2.unsqueeze(1)],dim=1)

            # choose 5:  cls_token
            if self.mode == "CLS":
                x = x0
            # choose 6:  Mean token
            if self.mode == "Mean":
                x = x2
        else:
            x = x

        x = x.reshape(B,-1,D)
        x = F.normalize(x, p=2, dim=2)  # intra-normalization
        x = x.view(B, -1)               # Flatten
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x
        
def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name], dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def get_backbone(args, pretrained_foundation, foundation_model_path):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = False
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct384"):
        backbone = cct_14_7x2_384(pretrained=False, progress=True, aggregation=None)
        checkpoint_path = '/data/users/cwy/0_datasets/model_weight/cct384/cct_14_7x2_384_imagenet.pth'
        if os.path.exists(checkpoint_path):
            print(f"Loading local CCT weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            keys_to_remove = ["classifier.fc.weight", "classifier.fc.bias"]
            for key in keys_to_remove:
                if key in state_dict:
                    del state_dict[key]
                    print(f"Ignored key: {key}")
            backbone.load_state_dict(state_dict, strict=True)
        for p in backbone.parameters():
            p.requires_grad = False
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vitb16_224"):
        backbone = ViTModel.from_pretrained('/data/users/cwy/0_datasets/model_weight/VPR_ViTB16-224')
        for p in backbone.parameters():
            p.requires_grad = False
        args.features_dim = 768
        return backbone
    elif args.backbone.startswith("vitb16_384"):
        backbone = ViTModel.from_pretrained('/data/users/cwy/0_datasets/model_weight/VPR_ViTB16-384')
        for p in backbone.parameters():
            p.requires_grad = False
        args.features_dim = 768
        return backbone
    elif args.backbone.startswith("dinov2_l"):
        backbone = vit_large(patch_size=14,img_size=518,init_values=1,block_chunks=0)
        if pretrained_foundation:
            assert foundation_model_path is not None, "Please specify foundation model path."
            model_dict = backbone.state_dict()
            state_dict = torch.load(foundation_model_path)
            model_dict.update(state_dict.items())
            backbone.load_state_dict(model_dict)
        for p in backbone.parameters():
            p.requires_grad = False
        args.features_dim = 1024
        return backbone
    elif args.backbone.startswith("dinov2_g"):
        backbone = vit_giant2(patch_size=14,img_size=518,init_values=1,block_chunks=0)
        if pretrained_foundation:
            assert foundation_model_path is not None, "Please specify foundation model path."
            model_dict = backbone.state_dict()
            state_dict = torch.load(foundation_model_path)
            model_dict.update(state_dict.items())
            backbone.load_state_dict(model_dict)
        for p in backbone.parameters():
            p.requires_grad = False
        args.features_dim = 1536
        return backbone
    elif args.backbone.startswith("dinov2"):
        backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
        if pretrained_foundation:
            assert foundation_model_path is not None, "Please specify foundation model path."
            model_dict = backbone.state_dict()
            state_dict = torch.load(foundation_model_path)
            model_dict.update(state_dict.items())
            backbone.load_state_dict(model_dict)

        for p in backbone.parameters():
                p.requires_grad = False

        # if args.freeze_te:
        #     for p in backbone.parameters():
        #         p.requires_grad = False
        #     for name, child in backbone.blocks.named_children():
        #         if int(name) >= args.freeze_te:
        #             for params in child.parameters():
        #                 params.requires_grad = True

        args.features_dim = 768 #1024
        return backbone
    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone

def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

