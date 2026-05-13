import os
import time
import uuid
import threading
import base64
import io
import wave

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import speech_recognition as sr
import cv2
from flask import Flask, render_template, request, jsonify

import pyaudio

from model import Net, apply_attention
from utils import get_transform
from resnet import resnet152, Bottleneck

from gtts import gTTS
from pydub import AudioSegment  # MP3 -> WAV


# 0. Torch 설정
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



# 1. Flask 앱 생성
app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/vqa_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# 2. 오디오 설정
print("[*] Available audio devices:")
audio_check = pyaudio.PyAudio()
for index in range(audio_check.get_device_count()):
    desc = audio_check.get_device_info_by_index(index)
    print(" DEVICE: {}, INDEX: {}, RATE: {}".format(
        desc["name"], index, int(desc["defaultSampleRate"])
    ))
audio_check.terminate()

MIC_DEVICE_ID = 11  # 필요시 None 로 바꾸면 default device
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
SAMPLE_SIZE = 2

recording_state = {
    "is_recording": False,
    "frames": [],
    "thread": None
}



# 3. VQA 설정

ORIGINAL_CHECKPOINT = "/home/jetson/Downloads/formina/logs/2017-08-04_00.55.19.pth"

FP16_CHECKPOINT = "/home/jetson/Downloads/formina/vqa_pruned_50_distilled_fp16.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

IMAGE_SIZE = 448
CENTRAL_FRACTION = 1.0
PRUNE_RATIO = 0.5



# 4. Vocab + FP16 state 로드
def load_fp16_checkpoint_and_vocab(fp16_path, original_ckpt_path=None):
    if not os.path.exists(fp16_path):
        raise FileNotFoundError(f"FP16 checkpoint not found: {fp16_path}")

    print(f"[*] Loading FP16 checkpoint from: {fp16_path}")
    t0 = time.time()
    obj = torch.load(fp16_path, map_location="cpu")
    print("[*] Done loading FP16 file. Took {:.2f} sec".format(time.time() - t0))

    vocab_local = None
    state_local = None

    if isinstance(obj, dict):
        if "weights" in obj:
            state_local = obj["weights"]
        elif "model_state_dict" in obj:
            state_local = obj["model_state_dict"]
        else:
            state_local = obj

        if "vocab" in obj:
            vocab_local = obj["vocab"]
    else:
        state_local = None

    if state_local is None and isinstance(obj, dict):
        state_local = obj

    if vocab_local is None:
        if original_ckpt_path and os.path.exists(original_ckpt_path):
            print("[*] FP16 checkpoint has no vocab. Falling back to ORIGINAL for vocab only.")
            saved_state = torch.load(original_ckpt_path, map_location="cpu")
            vocab_local = saved_state.get("vocab", None)
        else:
            vocab_local = None

    if vocab_local is None:
        raise RuntimeError(
            "Vocab not found in FP16 checkpoint and ORIGINAL_CHECKPOINT is unavailable.\n"
            "You need vocab to tokenize questions."
        )

    clean_state = {}
    if isinstance(state_local, dict):
        for k, v in state_local.items():
            clean_state[k.replace("module.", "")] = v
    else:
        raise RuntimeError("FP16 checkpoint format not supported.")

    return clean_state, vocab_local


# 로드 실행
CLEAN_FP16_STATE, vocab = load_fp16_checkpoint_and_vocab(FP16_CHECKPOINT, ORIGINAL_CHECKPOINT)

token_to_index = vocab["question"]
answer_to_index = vocab["answer"]
num_tokens = len(token_to_index) + 1

# idx -> answer
idx_to_answer = ["unk"] * len(answer_to_index)
for w, idx in answer_to_index.items():
    if idx < len(idx_to_answer):
        idx_to_answer[idx] = w



# 5. 이미지 캐시 (RAW만 유지)
cached_image_raw = None
cached_image_path = None



# 6. Structured Pruning 로직
def prune_bottleneck_layer(block, prune_ratio=0.5):
    w = block.conv1.weight.data
    importance = torch.sum(torch.abs(w), dim=(1, 2, 3))

    num_total = w.shape[0]
    num_keep = int(num_total * (1 - prune_ratio))
    if num_keep < 1:
        return

    keep_indices = torch.topk(importance, num_keep)[1]
    keep_indices, _ = torch.sort(keep_indices)

    # Conv1
    old_conv1 = block.conv1
    new_conv1 = nn.Conv2d(
        old_conv1.in_channels, num_keep,
        kernel_size=1, stride=old_conv1.stride, padding=0,
        bias=(old_conv1.bias is not None)
    )
    new_conv1.weight.data = old_conv1.weight.data[keep_indices]
    if old_conv1.bias is not None:
        new_conv1.bias.data = old_conv1.bias.data[keep_indices]

    # BN1
    old_bn1 = block.bn1
    new_bn1 = nn.BatchNorm2d(num_keep)
    new_bn1.weight.data = old_bn1.weight.data[keep_indices]
    new_bn1.bias.data = old_bn1.bias.data[keep_indices]
    new_bn1.running_mean = old_bn1.running_mean[keep_indices]
    new_bn1.running_var = old_bn1.running_var[keep_indices]

    # Conv2
    old_conv2 = block.conv2
    new_conv2 = nn.Conv2d(
        num_keep, num_keep,
        kernel_size=3, stride=old_conv2.stride, padding=1,
        bias=(old_conv2.bias is not None)
    )
    new_conv2.weight.data = old_conv2.weight.data[keep_indices][:, keep_indices]
    if old_conv2.bias is not None:
        new_conv2.bias.data = old_conv2.bias.data[keep_indices]

    # BN2
    old_bn2 = block.bn2
    new_bn2 = nn.BatchNorm2d(num_keep)
    new_bn2.weight.data = old_bn2.weight.data[keep_indices]
    new_bn2.bias.data = old_bn2.bias.data[keep_indices]
    new_bn2.running_mean = old_bn2.running_mean[keep_indices]
    new_bn2.running_var = old_bn2.running_var[keep_indices]

    # Conv3
    old_conv3 = block.conv3
    new_conv3 = nn.Conv2d(
        num_keep, old_conv3.out_channels,
        kernel_size=1, bias=(old_conv3.bias is not None)
    )
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



# 7. 모델 정의
class ResNetLayer4(nn.Module):
    def __init__(self):
        super().__init__()
        print("[*] Loading ResNet152 (pretrained=False)...")
        t0 = time.time()

        self.r_model = resnet152(pretrained=False)
        self.r_model.eval()

        print("[*] ResNet152 loaded. Took {:.2f} sec".format(time.time() - t0))

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
        answer = self.classifier(combined)
        return answer



# 8. 전처리 / 텍스트 인코딩
transform = get_transform(IMAGE_SIZE, central_fraction=CENTRAL_FRACTION)


def preprocess_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img).convert("RGB")
    elif isinstance(img, str):
        img = Image.open(img).convert("RGB")
    else:
        img = img.convert("RGB")

    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(device)


def encode_question(question):
    q = question.strip().lower().rstrip("?")
    tokens = q.split()
    if len(tokens) == 0:
        tokens = ["pad"]

    indices = [token_to_index.get(w, 0) for w in tokens]
    q_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    q_len = torch.LongTensor([len(indices)]).to(device)
    return q_tensor, q_len



# 9. 모델 메모리/파라미터 계산
def get_model_size_mb(model):
    mem_size = 0
    for param in model.parameters():
        mem_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        mem_size += buffer.nelement() * buffer.element_size()
    return mem_size / 1024 / 1024


def get_real_param_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    for m in model.modules():
        if isinstance(m, (nn.quantized.Conv2d, nn.quantized.Linear)):
            total_params += m.weight().numel()
            if m.bias() is not None:
                total_params += m.bias().numel()
    return total_params


def run_once_bench(
    model,
    img,
    q,
    q_len,
    label,
    override_dtype=None,
    warmup=5,
    repeat=20,
):
    model.eval()
    mem_mb = get_model_size_mb(model)
    param_count = get_real_param_count(model)

    times = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # 워밍업
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(img, q, q_len)
            torch.cuda.synchronize()

        # 반복 측정
        with torch.no_grad():
            for _ in range(repeat):
                starter.record()
                _ = model(img, q, q_len)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))

        # 결과/정답용 1회 실행
        with torch.no_grad():
            torch.cuda.synchronize()
            out = model(img, q, q_len)
            torch.cuda.synchronize()

    else:
        # 워밍업
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(img, q, q_len)

        # 반복 측정
        with torch.no_grad():
            for _ in range(repeat):
                t0 = time.time()
                _ = model(img, q, q_len)
                t1 = time.time()
                times.append((t1 - t0) * 1000.0)

        # 결과/정답용 1회 실행
        with torch.no_grad():
            out = model(img, q, q_len)

    # 후처리 (softmax, argmax, answer)
    with torch.no_grad():
        prob = F.softmax(out, dim=1)
        score, answer_idx = prob.max(dim=1)
        ans = idx_to_answer[answer_idx.item()] if answer_idx.item() < len(idx_to_answer) else "unk"

    speed_mean = float(np.mean(times)) if len(times) > 0 else float("nan")
    speed_std = float(np.std(times)) if len(times) > 0 else float("nan")

    print(f"\n[{label} RESULT]")
    print(f" ├── Answer : {ans}")
    print(f" ├── Confidence : {score.item():.4f}")
    print(f" ├── Speed (mean over {repeat}) : {speed_mean:.2f} ms/run (± {speed_std:.2f})")
    print(f" ├── Memory Size : {mem_mb:.2f} MB")
    print(f" └── Params : {param_count:,}")
    print("-" * 50)

    has_param = any(True for _ in model.parameters())
    dtype_str = "unknown"
    if override_dtype is not None:
        dtype_str = override_dtype
    elif has_param:
        try:
            dtype_str = str(next(model.parameters()).dtype)
        except StopIteration:
            dtype_str = "unknown"

    return {
        "label": label,
        "answer": ans,
        "confidence": float(score.item()),
        "speed_ms": speed_mean,
        "speed_std_ms": speed_std,
        "memory_mb": float(mem_mb),
        "params": int(param_count),
        "dtype": dtype_str
    }



# 10. state_dict 로드 유틸
def safe_load_state_dict(model, incoming_state):
    """shape가 맞는 파라미터만 골라서 로드"""
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

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[SafeLoad] Missing keys (partial): {len(missing)}")
    if unexpected:
        print(f"[SafeLoad] Unexpected keys (partial): {len(unexpected)}")



# 10-1. 전역 단일 모델
MODEL_SINGLE = None


def init_single_model():
    global MODEL_SINGLE

    print("[*] Initializing SINGLE VQA model from FP16 checkpoint...")

    # 1) 모델 생성
    model = VQAResNetModel(num_tokens)

    # 2) 아키텍처 맞추기 위해 pruning 적용
    model = model.to("cpu")
    pruned_count = apply_structured_pruning(model.resnet_layer4.r_model, ratio=PRUNE_RATIO)
    print(f"    [Pruning] Applied to {pruned_count} Bottleneck blocks (ratio={PRUNE_RATIO})")

    # 3) state 로드
    safe_load_state_dict(model, CLEAN_FP16_STATE)

    # 4) dtype/device 정리
    if device.type == "cuda":
        model = model.half().to(device).eval()
    else:
        model = model.float().to(device).eval()

    MODEL_SINGLE = model
    print("  ✔ SINGLE model ready on", device, "| dtype:", next(MODEL_SINGLE.parameters()).dtype)



# 11. 카메라 (stateless open/read/release)

LAST_CAMERA_SOURCE = None

GST_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def _try_capture_with_cap(cap, warmup=2):
    for _ in range(warmup):
        try:
            cap.read()
        except Exception:
            pass

    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    try:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def capture_from_camera(retries=3, warmup=2, sleep_between=0.08):
    global LAST_CAMERA_SOURCE

    # 1) CSI
    for attempt in range(1, retries + 1):
        cap = None
        try:
            cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
            if cap is not None and cap.isOpened():
                img = _try_capture_with_cap(cap, warmup=warmup)
                if img is not None:
                    LAST_CAMERA_SOURCE = "CSI"
                    return img
            print(f"[Camera][CSI] read failed (attempt {attempt}/{retries})")
        except Exception as e:
            print(f"[Camera][CSI] exception (attempt {attempt}/{retries}): {e}")
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

        time.sleep(sleep_between)

    # 2) USB fallback
    for attempt in range(1, retries + 1):
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap is not None and cap.isOpened():
                img = _try_capture_with_cap(cap, warmup=warmup)
                if img is not None:
                    LAST_CAMERA_SOURCE = "USB"
                    return img
            print(f"[Camera][USB] read failed (attempt {attempt}/{retries})")
        except Exception as e:
            print(f"[Camera][USB] exception (attempt {attempt}/{retries}): {e}")
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

        time.sleep(sleep_between)

    LAST_CAMERA_SOURCE = None
    print("[Camera] Error: Cannot capture from CSI/USB")
    return None



# 12. 오디오 녹음/STT
def get_new_recording_path():
    return os.path.join(UPLOAD_FOLDER, f"recording_{uuid.uuid4().hex}.wav")


def record_audio_thread():
    global recording_state
    p = pyaudio.PyAudio()

    stream_params = {
        "format": FORMAT,
        "channels": CHANNELS,
        "rate": RATE,
        "input": True,
        "frames_per_buffer": CHUNK
    }
    if MIC_DEVICE_ID is not None:
        stream_params["input_device_index"] = MIC_DEVICE_ID

    try:
        stream = p.open(**stream_params)
        print("[Audio] Recording started...")
        recording_state["frames"] = []

        while recording_state["is_recording"]:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                recording_state["frames"].append(data)
            except Exception as e:
                print("[Audio] Read error:", e)
                break

        print("[Audio] Recording stopped. Frames:", len(recording_state["frames"]))
        stream.stop_stream()
        stream.close()

    except Exception as e:
        print("[Audio] Error:", e)
    finally:
        p.terminate()


def save_wav(frames, filepath):
    wf = wave.open(filepath, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPLE_SIZE)
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    print("[Audio] Saved to:", filepath)


def audio_to_text(wav_path, retries=5):
    if not os.path.exists(wav_path):
        return None

    r = sr.Recognizer()

    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
    except Exception as e:
        print("[STT] Audio file error:", e)
        return None

    for attempt in range(retries + 1):
        try:
            print(f"[STT] Processing... (attempt {attempt + 1}/{retries + 1})")
            text = r.recognize_google(audio, language="en-US")
            print("[STT] Recognized:", text)
            return text
        except sr.UnknownValueError:
            print("[STT] Could not understand audio")
            return None
        except sr.RequestError as e:
            print("[STT] API error:", e)
            time.sleep(1)
        except Exception as e:
            print("[STT] Error:", e)
            time.sleep(1)

    return None



# 13. TTS (비동기, MP3->WAV)
def text_to_speech(text):
    try:
        temp_mp3 = f"/tmp/vqa_tts_{uuid.uuid4().hex}.mp3"
        output_wav = f"/tmp/vqa_tts_{uuid.uuid4().hex}.wav"

        tts = gTTS(text=text, lang="en")
        tts.save(temp_mp3)

        sound = AudioSegment.from_mp3(temp_mp3)
        sound.export(output_wav, format="wav")

        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

        print(f"[TTS] Created WAV: {output_wav}")
        return output_wav
    except Exception as e:
        print("[TTS] Error:", e)
        return None


def play_audio(filepath):
    try:
        wf = wave.open(filepath, "rb")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()

        time.sleep(0.05)
        if os.path.exists(filepath):
            os.remove(filepath)

    except Exception as e:
        print("[Playback Error]", e)


def async_tts(answer_text):
    if not answer_text.strip().endswith("?"):
        answer_text = answer_text.strip() + "?"

    def worker():
        path = text_to_speech(answer_text)
        if path:
            play_audio(path)

    threading.Thread(target=worker, daemon=True).start()



# 14. /ask 파이프라인 (SINGLE MODEL)
def get_cached_or_disk_image(image_path):
    global cached_image_raw, cached_image_path

    if cached_image_raw is not None and cached_image_path == image_path:
        return cached_image_raw

    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print("[Image] Disk load error:", e)
        return None


def run_single_model_for_question(image_path, question):
    global MODEL_SINGLE

    img_raw = get_cached_or_disk_image(image_path)
    if img_raw is None:
        return [], "Error: Image load failed"

    if MODEL_SINGLE is None:
        print("[WARN] MODEL_SINGLE not initialized. Calling init_single_model() lazily.")
        init_single_model()

    img = preprocess_image(img_raw)
    q_tensor, q_len = encode_question(question)

    results = []

    if MODEL_SINGLE is not None:
        print("\n--- [SINGLE] Pruned + Distilled + FP16 ---")

        if device.type == "cuda" and next(MODEL_SINGLE.parameters()).dtype == torch.float16:
            img_in = img.half()
            results.append(
                run_once_bench(
                    MODEL_SINGLE,
                    img_in,
                    q_tensor,
                    q_len,
                    "Pruned + Distilled + FP16 (Single)",
                    override_dtype="torch.float16",
                )
            )
        else:
            results.append(
                run_once_bench(
                    MODEL_SINGLE,
                    img,
                    q_tensor,
                    q_len,
                    "Pruned + Distilled (Single)",
                )
            )
    else:
        return [], "Error: Model init failed"

    final_answer = results[-1]["answer"] if results else "Error"
    return results, final_answer



# 15. Flask 라우트
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    global cached_image_raw, cached_image_path

    t0 = time.time()
    img = capture_from_camera()
    if img is None:
        return jsonify({"success": False, "error": "Camera error"})

    img_pil = Image.fromarray(img)

    temp_path = os.path.join(UPLOAD_FOLDER, "captured.jpg")
    img_pil.save(temp_path)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    cached_image_raw = img_pil
    cached_image_path = temp_path

    elapsed = time.time() - t0
    print(f"[Capture] Done in {elapsed:.2f}s (IO only, NO torch) | source={LAST_CAMERA_SOURCE}")

    return jsonify({
        "success": True,
        "image": "data:image/jpeg;base64," + img_base64,
        "path": temp_path,
        "elapsed": f"{elapsed:.2f}s"
    })


@app.route("/upload_image", methods=["POST"])
def upload_image():
    global cached_image_raw, cached_image_path

    t0 = time.time()
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"})

    temp_path = os.path.join(UPLOAD_FOLDER, "uploaded.jpg")
    file.save(temp_path)

    img_pil = Image.open(temp_path).convert("RGB")

    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    cached_image_raw = img_pil
    cached_image_path = temp_path

    elapsed = time.time() - t0
    print(f"[Upload] Done in {elapsed:.2f}s (IO only, NO torch)")

    return jsonify({
        "success": True,
        "image": "data:image/jpeg;base64," + img_base64,
        "path": temp_path,
        "elapsed": f"{elapsed:.2f}s"
    })


@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording_state

    if recording_state["is_recording"]:
        return jsonify({"success": False, "error": "Already recording"})

    recording_state["is_recording"] = True
    recording_state["frames"] = []
    recording_state["thread"] = threading.Thread(target=record_audio_thread)
    recording_state["thread"].start()

    return jsonify({"success": True, "message": "Recording started"})


@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    global recording_state

    if not recording_state["is_recording"]:
        return jsonify({"success": False, "error": "Not recording"})

    recording_state["is_recording"] = False
    if recording_state["thread"]:
        recording_state["thread"].join(timeout=2)

    frames = recording_state["frames"]
    if len(frames) == 0:
        return jsonify({"success": False, "error": "No audio recorded"})

    wav_path = get_new_recording_path()
    save_wav(frames, wav_path)

    text = audio_to_text(wav_path)

    try:
        os.remove(wav_path)
    except Exception:
        pass

    if text is None:
        return jsonify({"success": False, "error": "Speech recognition failed"})

    return jsonify({"success": True, "text": text})


@app.route("/recording_status", methods=["GET"])
def recording_status():
    return jsonify({
        "is_recording": recording_state["is_recording"],
        "frames_count": len(recording_state["frames"])
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    image_path = data.get("image_path")
    question = data.get("question")

    if not image_path or not os.path.exists(image_path):
        return jsonify({"success": False, "error": "No image"})
    if not question:
        return jsonify({"success": False, "error": "No question"})

    results, final_answer = run_single_model_for_question(image_path, question)

    async_tts(final_answer)

    return jsonify({
        "success": True,
        "question": question,
        "final_answer": final_answer,
        "results": results,  
        "device": str(device),
    })



# 16. HTML 템플릿 
def create_templates():
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(template_dir, exist_ok=True)

    html_content = """
<!DOCTYPE html>
<html>
<head>
  <title>VQA Demo - Single-Sample Bench</title>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: Arial, sans-serif;
      background: #1a1a2e;
      color: white;
      min-height: 100vh;
      padding: 20px;
    }
    h1 { text-align: center; margin-bottom: 8px; }
    .container {
      max-width: 980px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
    }
    .panel {
      flex: 1;
      min-width: 310px;
      background: #16213e;
      padding: 20px;
      border-radius: 14px;
    }
    h3 { margin-bottom: 12px; color: #eee; }

    .image-box {
      background: #000;
      border-radius: 10px;
      min-height: 250px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    .image-box img { max-width: 100%; max-height: 300px; }
    .image-placeholder { color: #666; text-align: center; }
    .image-badge {
      position: absolute; top: 10px; right: 10px;
      background: #4CAF50; padding: 5px 12px;
      border-radius: 20px; font-size: 12px;
      display:none;
    }
    .btn-row { display: flex; gap: 10px; margin-top: 10px; }
    button {
      flex: 1; padding: 12px; border: none; border-radius: 8px;
      cursor: pointer; font-size: 14px; font-weight: bold;
      transition: all 0.2s;
    }
    button:hover { opacity: 0.9; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-capture { background: #4CAF50; color: white; }
    .btn-upload { background: #2196F3; color: white; }
    .btn-reset  { background: #ff9800; color: white; }

    .perf-info { font-size: 11px; color: #aaa; text-align: center; margin-top: 6px; }

    .btn-record {
      background: #607D8B; color: white; width: 100%;
      padding: 14px; font-size: 16px;
    }
    .btn-record.recording {
      background: #f44336; animation: pulse 1s infinite;
    }
    .btn-record.processing { background: #ff9800; }
    @keyframes pulse {
      0%,100% { box-shadow:0 0 0 0 rgba(244,67,54,0.7) }
      50% { box-shadow:0 0 0 15px rgba(244,67,54,0) }
    }

    .record-status {
      text-align: center; padding: 10px; margin-top: 10px;
      border-radius: 8px; font-weight: bold;
    }
    .record-status.recording { background: #ffcdd2; color: #c62828; }
    .record-status.processing { background: #fff3e0; color: #e65100; }
    .record-status.success { background: #c8e6c9; color: #2e7d32; }
    .record-status.error { background: #ffcdd2; color: #c62828; }
    .record-status.idle { background: #eceff1; color: #607D8B; }

    .question-box {
      background: #0f3460; padding: 15px; border-radius: 10px;
      margin-top: 15px; border: 2px solid transparent;
    }
    .question-box.voice { border-color: #4CAF50; }
    .q-label { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .voice-tag {
      background: #4CAF50; padding: 3px 10px; border-radius: 15px;
      font-size: 11px; display: none;
    }
    .question-box.voice .voice-tag { display: inline; }

    input[type="text"] {
      width: 100%; padding: 12px; border: none; border-radius: 8px;
      font-size: 16px; background: #1a1a2e; color: white;
    }

    .status {
      text-align: center; padding: 10px; border-radius: 8px; margin: 10px 0;
    }
    .status.error { background: #ffcdd2; color: #c62828; }
    .status.processing { background: #fff3e0; color: #e65100; }
    .status.success { background: #c8e6c9; color: #2e7d32; }

    .btn-ask {
      background: linear-gradient(45deg, #667eea, #764ba2);
      color: white; width: 100%; padding: 15px; font-size: 18px;
      margin-top: 15px;
    }

    .result-card {
      background: #0f3460; padding: 14px; border-radius: 10px;
      margin-top: 10px; border: 1px solid #263159;
    }
    .result-title {
      font-weight: bold; font-size: 14px; margin-bottom: 6px;
      color: #cdd6ff;
    }
    .result-row {
      display: flex; justify-content: space-between;
      font-size: 12px; color: #ddd; margin: 2px 0;
    }
    .result-answer {
      font-size: 22px; font-weight: bold; color: #4CAF50;
      text-align: center; margin: 8px 0 2px 0;
    }

    .placeholder {
      text-align: center; padding: 40px; color: #666;
    }
  </style>
</head>
<body>
  <h1>Jetson nano demo</h1>
  <div class="container">
    <div class="panel">
      <h3>Image</h3>
      <div class="image-box" id="imageBox">
        <div class="image-placeholder" id="imagePlaceholder">
          <p style="font-size:40px;">📷</p>
          <p>Camera capture or Image upload</p>
        </div>
        <img id="previewImg" src="" style="display:none;">
        <div class="image-badge" id="imageBadge">✓ Selected</div>
      </div>

      <div class="btn-row" id="imageButtons">
        <button class="btn-capture" onclick="captureImage()">📸 Camera capture</button>
        <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="uploadImage(this)">
        <button class="btn-upload" onclick="document.getElementById('fileInput').click()">📁 Image upload</button>
      </div>
      <div class="btn-row" id="resetRow" style="display:none;">
        <button class="btn-reset" onclick="resetImage()">🔄 Reselect</button>
      </div>
      <div class="perf-info" id="imagePerf"></div>

      <h3 style="margin-top:25px;">🎤 Speech Recognition</h3>
      <button class="btn-record" id="recordBtn" onclick="toggleRecording()">🎙️ Start recording</button>
      <div class="record-status idle" id="recordStatus">Idle</div>

      <div class="question-box" id="qBox">
        <div class="q-label">
          <span>💬 Question</span>
          <span class="voice-tag">🎤 Audio</span>
        </div>
        <input type="text" id="qInput" placeholder="Enter a question...">
      </div>

      <div class="btn-row">
        <button onclick="setQ('What color is this?')">What color?</button>
        <button onclick="setQ('How many people?')">How many?</button>
      </div>

      <button class="btn-ask" onclick="askVQA()">Run Demo</button>
    </div>

    <div class="panel">
      <h3>Results</h3>
      <div id="mainStatus"></div>
      <div id="resultsArea"></div>
      <div class="placeholder" id="placeholder">
        <p style="font-size:50px;">🤔</p>
        <p>Input Image and Question</p>
      </div>
    </div>
  </div>

<script>
  var imagePath = null;
  var isRecording = false;

  function captureImage() {
    document.getElementById('recordBtn').disabled = true;
    showImageStatus('📸 Capturing..');
    document.getElementById('imagePerf').textContent = '';

    fetch('/capture', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        document.getElementById('recordBtn').disabled = false;
        if (data.success) {
          showImage(data.image, data.path);
          document.getElementById('imagePerf').textContent = 'Capture: ' + (data.elapsed || '');
        } else {
          alert('Camera error: ' + data.error);
          resetImagePlaceholder();
        }
      })
      .catch(e => {
        document.getElementById('recordBtn').disabled = false;
        alert('error: ' + e);
        resetImagePlaceholder();
      });
  }

  function uploadImage(input) {
    if (!input.files[0]) return;
    showImageStatus('📤 Uploading..');
    document.getElementById('imagePerf').textContent = '';

    var formData = new FormData();
    formData.append('image', input.files[0]);

    fetch('/upload_image', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          showImage(data.image, data.path);
          document.getElementById('imagePerf').textContent = 'Upload: ' + (data.elapsed || '');
        } else {
          alert('Upload error: ' + data.error);
          resetImagePlaceholder();
        }
      })
      .catch(e => {
        alert('error: ' + e);
        resetImagePlaceholder();
      });
  }

  function showImage(imgSrc, path) {
    imagePath = path;
    document.getElementById('previewImg').src = imgSrc;
    document.getElementById('previewImg').style.display = 'block';
    document.getElementById('imagePlaceholder').style.display = 'none';
    document.getElementById('imageBadge').style.display = 'block';
    document.getElementById('imageButtons').style.display = 'none';
    document.getElementById('resetRow').style.display = 'flex';
  }

  function resetImage() {
    imagePath = null;
    document.getElementById('previewImg').style.display = 'none';
    document.getElementById('imagePlaceholder').style.display = 'block';
    document.getElementById('imageBadge').style.display = 'none';
    document.getElementById('imageButtons').style.display = 'flex';
    document.getElementById('resetRow').style.display = 'none';
    document.getElementById('imagePerf').textContent = '';
    resetImagePlaceholder();
  }

  function showImageStatus(msg) {
    document.getElementById('imagePlaceholder').innerHTML = '<p>' + msg + '</p>';
  }

  function resetImagePlaceholder() {
    document.getElementById('imagePlaceholder').innerHTML =
      '<p style="font-size:40px;">📷</p><p>Camera capture or Image upload</p>';
  }

  function toggleRecording() {
    if (!isRecording) startRecording();
    else stopRecording();
  }

  function startRecording() {
    var btn = document.getElementById('recordBtn');
    var status = document.getElementById('recordStatus');

    btn.disabled = true;
    btn.textContent = 'Starting...';

    fetch('/start_recording', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          isRecording = true;
          btn.disabled = false;
          btn.textContent = 'Stop recording';
          btn.className = 'btn-record recording';
          status.textContent = 'Recording!';
          status.className = 'record-status recording';
        } else {
          btn.disabled = false;
          btn.textContent = 'Start recording';
          status.textContent = '❌ ' + data.error;
          status.className = 'record-status error';
        }
      })
      .catch(e => {
        btn.disabled = false;
        btn.textContent = 'Start recording';
        status.textContent = '❌ ' + e;
        status.className = 'record-status error';
      });
  }

  function stopRecording() {
    var btn = document.getElementById('recordBtn');
    var status = document.getElementById('recordStatus');

    btn.disabled = true;
    btn.textContent = 'Processing...';
    btn.className = 'btn-record processing';

    status.textContent = 'Recognizing audio';
    status.className = 'record-status processing';

    fetch('/stop_recording', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        isRecording = false;
        btn.disabled = false;
        btn.textContent = 'Start recording';
        btn.className = 'btn-record';

        if (data.success) {
          setQ(data.text, true);
          status.textContent = 'Recognized: "' + data.text + '"';
          status.className = 'record-status success';
        } else {
          status.textContent = '❌ ' + data.error;
          status.className = 'record-status error';
        }
      })
      .catch(e => {
        isRecording = false;
        btn.disabled = false;
        btn.textContent = 'Start recording';
        btn.className = 'btn-record';
        status.textContent = '❌ ' + e;
        status.className = 'record-status error';
      });
  }

  function setQ(text, fromVoice) {
    document.getElementById('qInput').value = text;
    document.getElementById('qBox').className = fromVoice ? 'question-box voice' : 'question-box';
  }

  document.getElementById('qInput').addEventListener('input', function() {
    document.getElementById('qBox').className = 'question-box';
  });

  function renderResults(results, deviceStr) {
    var area = document.getElementById('resultsArea');
    area.innerHTML = '';

    var dev = document.createElement('div');
    dev.className = 'status success';
    dev.textContent = 'Device: ' + deviceStr;
    area.appendChild(dev);

    results.forEach(r => {
      var card = document.createElement('div');
      card.className = 'result-card';

      var title = document.createElement('div');
      title.className = 'result-title';
      title.textContent = r.label + ' (' + r.dtype + ')';
      card.appendChild(title);

      var ans = document.createElement('div');
      ans.className = 'result-answer';
      ans.textContent = r.answer;
      card.appendChild(ans);

      var row1 = document.createElement('div');
      row1.className = 'result-row';
      row1.innerHTML = '<span>Confidence</span><span>' + r.confidence.toFixed(4) + '</span>';
      card.appendChild(row1);

      var row2 = document.createElement('div');
      row2.className = 'result-row';
      var speedText = r.speed_ms.toFixed(2) + ' ms/run';
      if (typeof r.speed_std_ms !== 'undefined' && !isNaN(r.speed_std_ms)) {
        speedText += ' (± ' + r.speed_std_ms.toFixed(2) + ')';
      }
      row2.innerHTML = '<span>Speed</span><span>' + speedText + '</span>';
      card.appendChild(row2);

      var row3 = document.createElement('div');
      row3.className = 'result-row';
      row3.innerHTML = '<span>Memory Size</span><span>' + r.memory_mb.toFixed(2) + ' MB</span>';
      card.appendChild(row3);

      var row4 = document.createElement('div');
      row4.className = 'result-row';
      row4.innerHTML = '<span>Params</span><span>' + r.params.toLocaleString() + '</span>';
      card.appendChild(row4);

      area.appendChild(card);
    });
  }

  function askVQA() {
    var question = document.getElementById('qInput').value;
    var mainStatus = document.getElementById('mainStatus');

    if (!imagePath) {
      mainStatus.innerHTML = '<div class="status error">Select an image</div>';
      return;
    }
    if (!question) {
      mainStatus.innerHTML = '<div class="status error">Enter a question</div>';
      return;
    }

    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('resultsArea').innerHTML = '';
    mainStatus.innerHTML = '<div class="status processing">Running...</div>';

    fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: imagePath, question: question })
    })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        mainStatus.innerHTML = '<div class="status success">Finished</div>';
        renderResults(data.results || [], data.device || '');
        setTimeout(function(){ mainStatus.innerHTML = ''; }, 1200);
      } else {
        mainStatus.innerHTML = '<div class="status error">❌ ' + data.error + '</div>';
      }
    })
    .catch(e => {
      mainStatus.innerHTML = '<div class="status error">❌ ' + e + '</div>';
    });
  }
</script>
</body>
</html>
    """

    with open(os.path.join(template_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    print("[*] Template created")



# 17. 메인

if __name__ == "__main__":
    print("=" * 60)
    print("VQA Demo Server (SINGLE MODEL)")
    print("=" * 60)
    print("✅ /capture, /upload_image: IO + base64 ONLY (NO torch, NO GPU)")
    print("✅ /ask: SINGLE model only (Pruned + Distilled + FP16)")
    print("✅ TTS: text_to_speech -> play_audio -> async_tts")
    print("✅ Camera: stateless open/read/release per /capture")
    print("✅ FP16_CHECKPOINT first, vocab fallback to ORIGINAL if needed")
    print("=" * 60)

    create_templates()

    try:
        init_single_model()
    except Exception as e:
        print("[ERROR] init_single_model failed:", e)

    print("\n[*] http://0.0.0.0:7860")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)