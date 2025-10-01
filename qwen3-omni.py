import os
import torch

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

if __name__ == '__main__':
    # vLLM engine v1 not supported yet
    # os.environ['VLLM_USE_V1'] = '0'

    MODEL_PATH = "models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    llm = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
            max_num_seqs=8,
            max_model_len=32768,
            seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "videos/delivery-1.mp4"}
            ], 
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": False,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)