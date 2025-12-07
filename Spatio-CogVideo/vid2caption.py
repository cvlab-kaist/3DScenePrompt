import os
PATH = '/mnt/cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

import io

import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from PIL import Image
import torch
import torchvision.transforms as T



MODEL_PATH = "THUDM/cogvlm2-llama3-caption"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16




def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)
    # video = video_data

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def load_video_from_images(base_path, name, num_frames=24, strategy='chat'):
    image_dir = os.path.join(base_path, name, 'color')
    image_files = sorted(os.listdir(image_dir))  # Ensure natural order
    total_frames = len(image_files)

    if strategy == 'base':
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        interval = max(1, total_frames // num_frames)
        frame_indices = [i * interval for i in range(num_frames)]
        frame_indices = [min(i, total_frames - 1) for i in frame_indices]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    transform = T.Compose([
        T.ToTensor(),  # Converts to [C, H, W] with range [0, 1]
    ])

    frames = []
    for idx in frame_indices:
        image_path = os.path.join(image_dir, image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        frames.append(image_tensor)

    video_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
    return video_tensor

def test(args):
    prompt = "Please describe this video in detail."
    temperature = 0.1
    # video_data = open('test.mp4', 'rb').read()
    from natsort import natsorted
    from tqdm import tqdm
    
    # base_path = '/mnt/DAVIS-data/megasam'
    # output_base_path = '/mnt/DAVIS-data/caption'
    # video_base_path = '/mnt/DAVIS-data/ours_continuous_gen/video'
    # base_path = '/mnt/DL3DV/validation/vggt_80_frame'
    # video_base_path = '/mnt/DL3DV/validation/videos'
    # output_base_path = '/mnt/DL3DV/validation/caption'
    # captions_path = '/mnt/DL3DV/validation/captions.txt'
    
    # base_path = '/mnt/RealEstate10K_Downloader/train_1_GT_frame/image'
    # video_base_path = '/mnt/dynpose-100k/train_9_GT_frame/video'
    # output_base_path = '/mnt/dynpose-100k/train_9_GT_frame/captions'
    # captions_path = '/mnt/DL3DV/validation/captions.txt'
    video_base_path = args.video_base_path
    output_base_path = args.output_base_path
    
    names = natsorted(os.listdir(video_base_path))
    names = [name.split('.')[0] for name in names if name.endswith('.mp4')]
    if args.batch!=  -1:
        data_len = len(names)
        # breakpoint()
        names = names[int(args.batch/args.total_batch* data_len):int((args.batch+1)/args.total_batch* data_len)]
    os.makedirs(output_base_path, exist_ok=True)
    # for name in names:
    captions = []
    for name in tqdm(names):
        # try:
            output_path = os.path.join(output_base_path, f'{name}.txt')
            if os.path.exists(output_path):
                print(f"Caption for {name} already exists. Skipping...")
                continue
            # images = load_video_from_images(base_path, name, num_frames=24, strategy='chat')
            # 
            # response = predict(prompt, images, temperature)
            
            video_name = f'{name}.mp4'
            video_path = os.path.join(video_base_path, video_name)
            video_data = open(video_path, 'rb').read()
            breakpoint()
            response = predict(prompt,video_data, temperature)
            # print(response)
            
            with open(output_path, 'w') as f:
                f.write(response)
                
            captions.append(f'{response}')
        # except:pass
            
    # with open(captions_path, 'w') as f:
    #     f.write('\n'.join(captions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
    parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
    parser.add_argument('--batch', type=int, default=-1, help='Current batch index for splitting dataset')
    parser.add_argument('--total_batch', type=int, default=10, help='Total number of batch splits')
    parser.add_argument('--video_base_path', type=str, default='', help='Path to the directory containing video files')
    parser.add_argument('--output_base_path', type=str, default='', help='Path to the directory to save captions')
    args = parser.parse_args()
    test(args)
