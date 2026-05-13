import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 기존 모듈
from model import Net, apply_attention
from resnet import resnet152, Bottleneck


#GPU 최적화 설정
BASE_DIR = '/root/pytorch-vqa'
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'val2014')
Q_PATH = os.path.join(DATA_DIR, 'v2_OpenEnded_mscoco_val2014_questions.json')
A_PATH = os.path.join(DATA_DIR, 'v2_mscoco_val2014_annotations.json')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'logs/2017-08-04_00.55.19.pth')

BATCH_SIZE = 64  
EPOCHS = 5
LR = 1e-4

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Available GPUs: {torch.cuda.device_count()}")
print(f"[*] Current Device: {device}")


# 1) 데이터셋 (이전과 동일)
class VQAValImageDataset(Dataset):
    def __init__(self, img_dir, q_path, a_path, vocab_from_ckpt, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.vocab = vocab_from_ckpt
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']
        
        with open(q_path, 'r') as f:
            self.questions = json.load(f)['questions']
        with open(a_path, 'r') as f:
            self.annotations = json.load(f)['annotations']
            
        self.qid2ann = {ann['question_id']: ann for ann in self.annotations}

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q_item = self.questions[idx]
        img_id = q_item['image_id']
        question_str = q_item['question']
        q_tokens = question_str.lower().rstrip("?").split()
        
        q_indices = [self.token_to_index.get(w, 0) for w in q_tokens]
        max_q_len = 30
        q_tensor = torch.zeros(max_q_len, dtype=torch.long)
        indices = torch.tensor(q_indices)[:max_q_len]
        q_tensor[:len(indices)] = indices
        q_len = len(indices)

        img_filename = f"COCO_val2014_{img_id:012d}.jpg"
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (448, 448))

        if self.transform:
            image = self.transform(image)
            
        ans_idx = 0
        if q_item['question_id'] in self.qid2ann:
            ann = self.qid2ann[q_item['question_id']]
            ans_str = ann['multiple_choice_answer']
            ans_idx = self.answer_to_index.get(ans_str, 0)
        
        return image, q_tensor, q_len, ans_idx


# 2) Pruning 함수 (0.5 Uniform)
def prune_bottleneck_layer(block, prune_ratio=0.5):
    w = block.conv1.weight.data
    importance = torch.sum(torch.abs(w), dim=(1, 2, 3))
    num_total = w.shape[0]
    num_keep = int(num_total * (1 - prune_ratio))
    if num_keep < 1: return
    
    _, keep_indices = torch.topk(importance, num_keep)
    keep_indices, _ = torch.sort(keep_indices)
    
    old_conv1 = block.conv1
    new_conv1 = nn.Conv2d(old_conv1.in_channels, num_keep, kernel_size=1, stride=old_conv1.stride, padding=0, bias=(old_conv1.bias is not None))
    new_conv1.weight.data = old_conv1.weight.data[keep_indices]
    if old_conv1.bias is not None: new_conv1.bias.data = old_conv1.bias.data[keep_indices]
    
    old_bn1 = block.bn1
    new_bn1 = nn.BatchNorm2d(num_keep)
    new_bn1.weight.data = old_bn1.weight.data[keep_indices]
    new_bn1.bias.data = old_bn1.bias.data[keep_indices]
    new_bn1.running_mean = old_bn1.running_mean[keep_indices]
    new_bn1.running_var = old_bn1.running_var[keep_indices]

    old_conv2 = block.conv2
    new_conv2 = nn.Conv2d(num_keep, num_keep, kernel_size=3, stride=old_conv2.stride, padding=1, bias=(old_conv2.bias is not None))
    temp = old_conv2.weight.data[keep_indices]
    new_conv2.weight.data = temp[:, keep_indices]
    if old_conv2.bias is not None: new_conv2.bias.data = old_conv2.bias.data[keep_indices]

    old_bn2 = block.bn2
    new_bn2 = nn.BatchNorm2d(num_keep)
    new_bn2.weight.data = old_bn2.weight.data[keep_indices]
    new_bn2.bias.data = old_bn2.bias.data[keep_indices]
    new_bn2.running_mean = old_bn2.running_mean[keep_indices]
    new_bn2.running_var = old_bn2.running_var[keep_indices]

    old_conv3 = block.conv3
    new_conv3 = nn.Conv2d(num_keep, old_conv3.out_channels, kernel_size=1, bias=(old_conv3.bias is not None))
    new_conv3.weight.data = old_conv3.weight.data[:, keep_indices]
    if old_conv3.bias is not None: new_conv3.bias.data = old_conv3.bias.data

    block.conv1 = new_conv1
    block.bn1 = new_bn1
    block.conv2 = new_conv2
    block.bn2 = new_bn2
    block.conv3 = new_conv3

def apply_structured_pruning(model, ratio=0.5):
    print(f"[*] Applying Structured Pruning (Uniform Ratio: {ratio})...")
    count = 0
    for m in model.modules():
        if isinstance(m, Bottleneck):
            prune_bottleneck_layer(m, ratio)
            count += 1
    print(f"[*] Pruned {count} blocks.")


# 2) 모델 정의 
class ResNetLayer4(nn.Module):
    def __init__(self):
        super().__init__()
        # GPU 학습을 위해 pretrained 모델 로드
        self.r_model = resnet152(pretrained=True)

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
    def __init__(self, num_tokens):
        super().__init__(num_tokens)
        self.resnet_layer4 = ResNetLayer4()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)
        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    )
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * hard_loss + (1. - alpha) * (T * T) * soft_loss


# 4) 메인 실행부
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 1. Vocab Load
    print(f"[*] Loading Checkpoint from {CHECKPOINT_PATH}...")
    saved_state = torch.load(CHECKPOINT_PATH, map_location='cpu')
    vocab_from_ckpt = saved_state['vocab']
    num_tokens = len(vocab_from_ckpt['question']) + 1

    # 2. Teacher 준비
    print("[*] Setting up Teacher...")
    teacher = VQAResNetModel(num_tokens)
    
    base_net = Net(num_tokens)
    clean_state = {k.replace("module.", ""): v for k, v in saved_state["weights"].items()}
    base_net.load_state_dict(clean_state)
    
    t_state = teacher.state_dict()
    t_state.update(base_net.state_dict())
    teacher.load_state_dict(t_state)
    
    # Multi-GPU 설정
    if torch.cuda.device_count() > 1:
        print(f"[*] Activate DataParallel on {torch.cuda.device_count()} GPUs!")
        teacher = nn.DataParallel(teacher)
    
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
        
    # 3. Student 준비 (Pruning 먼저 하고 DataParallel)
    print("[*] Setting up Student & Pruning 50%...")
    student = VQAResNetModel(num_tokens)
    student.load_state_dict(t_state) 
    
    apply_structured_pruning(student.resnet_layer4.r_model, ratio=0.5)
    
    # Multi-GPU 설정 (Student)
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)

    student.to(device)
    student.train()
    
    # 4. Dataset
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    print("[*] Loading Validation Dataset...")
    train_dataset = VQAValImageDataset(IMG_DIR, Q_PATH, A_PATH, vocab_from_ckpt, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    
    # 5. Training
    optimizer = optim.Adam(student.parameters(), lr=LR)
    
    print(f"[*] Starting Distillation for {EPOCHS} epochs on {device}...")
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        for i, (imgs, qs, q_lens, ans) in enumerate(train_loader):
            imgs, qs, q_lens, ans = imgs.to(device), qs.to(device), q_lens.to(device), ans.to(device)
            
            # Forward
            with torch.no_grad():
                t_logits = teacher(imgs, qs, q_lens)
            
            s_logits = student(imgs, qs, q_lens)
            
            loss = distillation_loss(s_logits, t_logits, ans, T=4.0, alpha=0.5)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"==> End of Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Time: {time.time()-start_time:.1f}s")
        
        model_to_save = student.module if hasattr(student, 'module') else student
        torch.save(model_to_save.state_dict(), f"vqa_pruned_50_distilled_epoch{epoch+1}.pth")

    print("[*] All Done. Best model saved.")

if __name__ == "__main__":
    main()