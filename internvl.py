from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Local InternVL3.5 model directory (HF format with remote code)
model_path = "models/OpenGVLab/InternVL3_5-30B-A3B"

# Example video. Replace with your own path if needed.
video_path = "videos/delivery-1.mp4"

# Initialize vLLM with multimodal enabled for video
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.99,
    trust_remote_code=True,
    limit_mm_per_prompt={"video": 1},
)

sampling_params = SamplingParams(max_tokens=1024)

# Build chat messages using the model's chat template (supports <video> tag)
video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the video in detail."},
            {"type": "video", "video": video_path},
        ],
    },
]

# Prepare prompt via the model's chat template
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
prompt = processor.apply_chat_template(
    video_messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Collect multimodal inputs (here only video). vLLM expects a per-modality list.
video_inputs = []
for msg in video_messages:
    content = msg.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                vpath = item.get("video")
                if vpath:
                    video_inputs.append(vpath)

mm_data = {}
if video_inputs:
    mm_data["video"] = video_inputs

# vLLM input payload
llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

# Generate and print the text output
outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
for o in outputs:
    print(o.outputs[0].text)
