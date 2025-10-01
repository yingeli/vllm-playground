from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import cv2
import numpy as np

# Local InternVL3.5 model directory (HF format with remote code)
model_path = "models/OpenGVLab/InternVL3_5-30B-A3B"

# Example video. Replace with your own path if needed.
video_path = "videos/delivery-1.mp4"

# Initialize vLLM with multimodal enabled for video
# Initialize vLLM with CPU-friendly settings if GPU is unavailable
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    enforce_eager=True,
    # Limit images per prompt; we will treat video as multiple images (frames)
    limit_mm_per_prompt={"image": 8},
)

sampling_params = SamplingParams(max_tokens=512)

# Build chat messages using the model's chat template (we will insert <image> tokens per frame)
messages_base = [
    {"role": "system", "content": "You are a helpful assistant."},
]

# We will prepare the prompt via the model's chat template after composing messages

# Collect multimodal inputs (here only video). vLLM expects a per-modality list.
def load_video_frames(path: str, max_frames: int = 8) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    if cap.isOpened():
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1
        cap.release()
    return np.stack(frames) if frames else np.empty((0,))

video_inputs = []
# Read frames from the example video
frames_np = load_video_frames(video_path, max_frames=8)
if frames_np.size != 0:
    for i in range(frames_np.shape[0]):
        video_inputs.append(frames_np[i])

mm_data = {}
if video_inputs:
    mm_data["image"] = video_inputs

# Compose final messages: user text + one image placeholder per frame
image_placeholders = [{"type": "image", "image": None} for _ in video_inputs]
messages = messages_base + [{
    "role": "user",
    "content": [{"type": "text", "text": "Describe the video in detail."}] + image_placeholders,
}]

# vLLM input payload
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

# Generate and print the text output
outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
for o in outputs:
    print(o.outputs[0].text)
