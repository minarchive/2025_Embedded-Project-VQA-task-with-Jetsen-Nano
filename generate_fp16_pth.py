import os
import torch
import torch.nn as nn
from resnet import resnet152, Bottleneck
from model import Net, apply_attention

ORIGINAL_CHECKPOINT = "C:\\Users\\mh050\\Desktop\\EAI_Final_Team_7\\pytorch-vqa\\logs\\2017-08-04_00.55.19.pth"
DISTILLED_CHECKPOINT = "C:\\Users\\mh050\\Desktop\\EAI_Final_Team_7\\vqa_pruned_50_distilled_epoch5.pth"
FP16_OUTPUT = "C:\\Users\\mh050\\Desktop\\EAI_Final_Team_7\\vqa_pruned_50_distilled_fp16.pth"
PRUNE_RATIO = 0.5


def prune_bottleneck_layer(block, prune_ratio=0.5):
    w = block.conv1.weight.data
    importance = torch.sum(torch.abs(w), dim=(1, 2, 3))
    num_total = w.shape[0]
    num_keep = int(num_total * (1 - prune_ratio))
    if num_keep < 1:
        return

    keep_indices = torch.topk(importance, num_keep)[1]
    keep_indices, _ = torch.sort(keep_indices)

    old_conv1 = block.conv1
    new_conv1 = nn.Conv2d(old_conv1.in_channels, num_keep, 1, stride=old_conv1.stride,
                          padding=0, bias=(old_conv1.bias is not None))
    new_conv1.weight.data = old_conv1.weight.data[keep_indices]
    if old_conv1.bias is not None:
        new_conv1.bias.data = old_conv1.bias.data[keep_indices]

    old_bn1 = block.bn1
    new_bn1 = nn.BatchNorm2d(num_keep)
    new_bn1.weight.data = old_bn1.weight.data[keep_indices]
    new_bn1.bias.data = old_bn1.bias.data[keep_indices]
    new_bn1.running_mean = old_bn1.running_mean[keep_indices]
    new_bn1.running_var = old_bn1.running_var[keep_indices]

    old_conv2 = block.conv2
    new_conv2 = nn.Conv2d(num_keep, num_keep, 3, stride=old_conv2.stride, padding=1,
                          bias=(old_conv2.bias is not None))
    new_conv2.weight.data = old_conv2.weight.data[keep_indices][:, keep_indices]
    if old_conv2.bias is not None:
        new_conv2.bias.data = old_conv2.bias.data[keep_indices]

    old_bn2 = block.bn2
    new_bn2 = nn.BatchNorm2d(num_keep)
    new_bn2.weight.data = old_bn2.weight.data[keep_indices]
    new_bn2.bias.data = old_bn2.bias.data[keep_indices]
    new_bn2.running_mean = old_bn2.running_mean[keep_indices]
    new_bn2.running_var = old_bn2.running_var[keep_indices]

    old_conv3 = block.conv3
    new_conv3 = nn.Conv2d(num_keep, old_conv3.out_channels, 1,
                          bias=(old_conv3.bias is not None))
    new_conv3.weight.data = old_conv3.weight.data[:, keep_indices]
    if old_conv3.bias is not None:
        new_conv3.bias.data = old_conv3.bias.data

    block.conv1 = new_conv1
    block.bn1 = new_bn1
    block.conv2 = new_conv2
    block.bn2 = new_bn2
    block.conv3 = new_conv3

def apply_structured_pruning(model, ratio=0.5):
    count = 0
    for m in model.modules():
        if isinstance(m, Bottleneck):
            prune_bottleneck_layer(m, ratio)
            count += 1
    return count

class ResNetLayer4(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_model = resnet152(pretrained=True)
        self.r_model.eval()

    def forward(self, x):
        x = self.r_model.conv1(x)
        x = self.r_model.bn1(x)
        x = self.r_model.relu(x)
        x = self.r_model.maxpool(x)
        x = self.r_model.layer1(x)
        x = self.r_model.layer2(x)
        x = self.r_model.layer3(x)
        x = self.r_model.layer4(x)
        return x

class VQAResNetModel(Net):
    def __init__(self, embedding_tokens):
        super().__init__(embedding_tokens)
        self.resnet_layer4 = ResNetLayer4()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)
        combined = torch.cat([v, q], dim=1)
        return self.classifier(combined)

def safe_load_state_dict(model, incoming_state):
    if incoming_state is None:
        return
    model_state = model.state_dict()
    filtered = {}
    for k, v in incoming_state.items():
        if k in model_state and hasattr(model_state[k], "shape") and hasattr(v, "shape"):
            if model_state[k].shape == v.shape:
                filtered[k] = v
        elif k in model_state:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)

def main():
    if not os.path.exists(ORIGINAL_CHECKPOINT):
        raise FileNotFoundError(f"Original checkpoint not found: {ORIGINAL_CHECKPOINT}")
    if not os.path.exists(DISTILLED_CHECKPOINT):
        raise FileNotFoundError(f"Distilled checkpoint not found: {DISTILLED_CHECKPOINT}")

    print("[*] Loading original checkpoint...")
    original = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu")
    vocab = original["vocab"]
    clean_original = {k.replace("module.", ""): v for k, v in original["weights"].items()}
    num_tokens = len(vocab["question"]) + 1

    print("[*] Loading distilled checkpoint...")
    distil_obj = torch.load(DISTILLED_CHECKPOINT, map_location="cpu")
    if isinstance(distil_obj, dict) and "model_state_dict" in distil_obj:
        clean_distilled = {k.replace("module.", ""): v for k, v in distil_obj["model_state_dict"].items()}
    elif isinstance(distil_obj, dict):
        clean_distilled = {k.replace("module.", ""): v for k, v in distil_obj.items()}
    else:
        clean_distilled = None

    print("[*] Building pruned+distilled model...")
    model = VQAResNetModel(num_tokens)
    safe_load_state_dict(model, clean_original)
    apply_structured_pruning(model.resnet_layer4.r_model, ratio=PRUNE_RATIO)
    safe_load_state_dict(model, clean_distilled)

    print("[*] Casting to FP16 and exporting...")
    model = model.half()
    fp16_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(fp16_state, FP16_OUTPUT)
    print(f"[✓] Saved FP16 checkpoint to: {FP16_OUTPUT}")

if __name__ == "__main__":
    main()
