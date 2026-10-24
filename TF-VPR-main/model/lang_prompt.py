"""Natural-language prompt bank for training-free VPR (NLPrompt-inspired).

A frozen CLIP text encoder turns a bank of place/scene descriptions into text
prototypes.  MaskCLIP-style patch-text affinities then provide a semantic
salience map for the DINOv2 patch tokens.  No training and no labels are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Scene / structure / appearance prompts.  Kept generic so the bank transfers
# across cities; they act as semantic anchors rather than class labels.
DEFAULT_PROMPTS = [
    "a photo of a road", "a photo of a street", "a photo of a sidewalk",
    "a photo of a building", "a photo of a house", "a photo of a skyscraper",
    "a photo of a tree", "a photo of vegetation", "a photo of a park",
    "a photo of the sky", "a photo of clouds", "a photo of water",
    "a photo of a river", "a photo of the sea", "a photo of a bridge",
    "a photo of a tunnel", "a photo of a crosswalk", "a photo of a traffic sign",
    "a photo of a traffic light", "a photo of a car", "a photo of a bus",
    "a photo of a person", "a photo of a shop", "a photo of a fence",
    "a photo of a wall", "a photo of stairs", "a photo of a square",
    "a photo of a rural landscape", "a photo of an urban scene",
    "a photo of a night scene", "a photo of a rainy scene",
    "a photo of a snowy scene", "a photo of a sunny scene",
]


class LanguagePromptBank(nn.Module):
    def __init__(self, model_dir, prompts=None, clip_size=224):
        super().__init__()
        self.clip_size = clip_size
        self.clip = CLIPModel.from_pretrained(model_dir)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False
        self.tokenizer = CLIPTokenizer.from_pretrained(model_dir)
        prompts = prompts or DEFAULT_PROMPTS
        self.prompts = prompts
        tokens = self.tokenizer(prompts, padding=True, truncation=True,
                                max_length=77, return_tensors="pt")
        with torch.no_grad():
            text = self.clip.get_text_features(input_ids=tokens["input_ids"],
                                               attention_mask=tokens["attention_mask"])
        # transformers >=5 returns a model-output object rather than a tensor
        if not torch.is_tensor(text):
            text = text.pooler_output
        self.register_buffer("text_emb", F.normalize(text, dim=-1))

    def _to_clip_space(self, x):
        """x is ImageNet-normalised (B,3,H,W)."""
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        cmean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        cstd = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = x * std + mean
        return (x - cmean) / cstd

    @torch.no_grad()
    def patch_affinity(self, x, out_grid=None):
        """Return (B, T, H, W) cosine affinities between patch tokens and prompts."""
        x = self._to_clip_space(x)
        if x.shape[-1] != self.clip_size or x.shape[-2] != self.clip_size:
            x = F.interpolate(x, size=(self.clip_size, self.clip_size),
                              mode="bicubic", align_corners=False)
        out = self.clip.vision_model(pixel_values=x)
        patch = out.last_hidden_state[:, 1:, :]          # (B, Np, 768)
        patch = self.clip.visual_projection(patch)        # (B, Np, 512)
        patch = F.normalize(patch, dim=-1)
        sim = torch.einsum("bnd,td->bnt", patch, self.text_emb)  # (B,Np,T)
        B, Np, T = sim.shape
        s = int(round(Np ** 0.5))
        sim = sim.reshape(B, s, s, T).permute(0, 3, 1, 2)
        if out_grid is not None and (s != out_grid):
            sim = F.interpolate(sim, size=(out_grid, out_grid),
                                mode="bilinear", align_corners=False)
        return sim

    def semantic_tokens(self, images, tokens, prompt_ids, temp=0.1):
        """Language-anchored tokens: each prompt pools the patch tokens it matches.

        The resulting K tokens are new descriptor components (unlike a per-token
        gate, which is cancelled by the per-patch L2 normalisation inside CAM).
        """
        B, L, D = tokens.shape
        g = int(round(L ** 0.5))
        aff = self.patch_affinity(images, out_grid=g)[:, prompt_ids]   # B,K,g,g
        a = aff.reshape(B, len(prompt_ids), L)
        a = F.softmax(a / temp, dim=-1)
        out = torch.einsum("bkl,bld->bkd", a, tokens)
        return F.normalize(out, dim=-1)

    def token_gate(self, x, out_grid, gamma=0.3):
        """Semantic salience gate in [1-gamma, 1+gamma], shape (B, H*W, 1)."""
        aff = self.patch_affinity(x, out_grid=out_grid)       # B,T,H,W
        sal = aff.max(dim=1).values                            # B,H,W
        sal = sal.flatten(1)                                   # B,H*W
        sal = (sal - sal.mean(dim=1, keepdim=True)) / (sal.std(dim=1, keepdim=True) + 1e-6)
        return 1.0 + gamma * torch.tanh(sal).unsqueeze(-1)
