# test_attention_dict.py — same as test_attention but using dict inputs/expected/outputs
import torch

def init_once():
    """Called once on first load. Returns a state dict."""
    q = torch.randn(2, 8, 128, 64, device="cuda")
    k = torch.randn(2, 8, 128, 64, device="cuda")
    v = torch.randn(2, 8, 128, 64, device="cuda")
    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    weights = torch.softmax(scores, dim=-1)
    attn = weights @ v
    return {
        "inputs": {"q": q, "k": k, "v": v},
        "expected": {"attn": attn, "weights": weights},
    }

def run(inputs):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    weights = torch.softmax(scores, dim=-1)
    return {"attn": weights @ v, "weights": weights}
