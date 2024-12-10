import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
from gradio import update  # Added this import
import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils
from prompt_manager import PromptLibrary

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

device = 'cuda'
output_dir = Path('./output/gradio')

setup_eval_logging()

def get_model(variant: str, full_precision: bool = False) -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    """Load model and feature utils from configuration."""
    if variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {variant}')
    
    model: ModelConfig = all_model_cfg[variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    dtype = torch.float32 if full_precision else torch.bfloat16

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path)
    feature_utils = feature_utils.to(device, dtype).eval()

    return net, feature_utils, seq_cfg

# Initialize with default model
current_model = {
    'variant': 'large_44k_v2',
    'full_precision': False
}

net, feature_utils, seq_cfg = get_model(current_model['variant'], current_model['full_precision'])

def load_prompt_library():
    """Load the prompt library from the prompts file."""
    library_path = Path(__file__).parent / "./prompts/prompts.json"
    return PromptLibrary(library_path)

def update_subcategories(category):
    """Update subcategories when category changes."""
    library = load_prompt_library()
    subcats = library.get_subcategories(category)
    return gr.update(
        choices=[{"label": name, "value": id} for id, name in subcats],
        value=None,
        visible=True
    )

def update_prompts(category, subcategory):
    """Update available prompts based on category/subcategory selection."""
    if not category or not subcategory:
        return gr.update(choices=None, value=None)
    library = load_prompt_library()
    choices = library.get_prompt_choices(category, subcategory)
    return gr.update(
        choices=choices,
        value=choices[0] if choices else None,
        visible=True
    )

def update_negative_prompt(category, subcategory, prompt):
    """Get matching negative prompt for selected prompt."""
    library = load_prompt_library()
    return library.get_negative_prompt(category, subcategory, prompt)

def reload_model(variant: str, full_precision: bool):
    """Reload model if configuration changes."""
    global net, feature_utils, seq_cfg
    if variant != current_model['variant'] or full_precision != current_model['full_precision']:
        current_model['variant'] = variant
        current_model['full_precision'] = full_precision
        net, feature_utils, seq_cfg = get_model(variant, full_precision)
        log.info(f'Reloaded model: {variant} (full_precision={full_precision})')

@torch.inference_mode()
def video_to_audio(
    video: gr.Video,
    prompt_category: str,
    prompt_subcategory: str,
    prompt: str,
    custom_prompt: str,
    custom_negative: str,
    variant: str,
    seed: int,
    num_steps: int,
    cfg_strength: float,
    duration: float,
    mask_away_clip: bool,
    full_precision: bool,
    skip_video_composite: bool,
    quality: str = "normal"
):
    """Generate audio from video with enhanced configuration options."""
    # Use custom prompt if provided, otherwise use selected prompt
    final_prompt = custom_prompt if custom_prompt.strip() else prompt
    final_negative = custom_negative if custom_negative.strip() else update_negative_prompt(prompt_category, prompt_subcategory, prompt)
    
    reload_model(variant, full_precision)
    
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    
    # Adjust steps based on quality setting
    actual_steps = num_steps * 2 if quality == "high" else num_steps
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=actual_steps)

    clip_frames, sync_frames, duration = load_video(video, duration)
    if mask_away_clip:
        clip_frames = None
    else:
        clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [final_prompt],
                      negative_text=[final_negative],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]

    # Generate timestamp and paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_path = output_dir / f'{timestamp}.flac'
    video_path = output_dir / f'{timestamp}.mp4'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    torchaudio.save(audio_path, audio, seq_cfg.sampling_rate)
    
    if not skip_video_composite:
        make_video(video,
                   video_path,
                   audio,
                   sampling_rate=seq_cfg.sampling_rate,
                   duration_sec=seq_cfg.duration)
        return video_path
    else:
        return audio_path

@torch.inference_mode()
def text_to_audio(
    prompt_category: str,
    prompt_subcategory: str,
    prompt: str,
    custom_prompt: str,
    custom_negative: str,
    variant: str,
    seed: int,
    num_steps: int,
    cfg_strength: float,
    duration: float,
    full_precision: bool,
    quality: str = "normal"
):
    """Generate audio from text with enhanced configuration options."""
    # Use custom prompt if provided, otherwise use selected prompt
    final_prompt = custom_prompt if custom_prompt.strip() else prompt
    final_negative = custom_negative if custom_negative.strip() else update_negative_prompt(prompt_category, prompt_subcategory, prompt)
    
    reload_model(variant, full_precision)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    
    # Adjust steps based on quality setting
    actual_steps = num_steps * 2 if quality == "high" else num_steps
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=actual_steps)

    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [final_prompt],
                      negative_text=[final_negative],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    audio_path = output_dir / f'{timestamp}.flac'
    torchaudio.save(audio_path, audio, seq_cfg.sampling_rate)
    return audio_path

# Model variants available
model_choices = list(all_model_cfg.keys())

# Create initial choices for dropdown
prompt_library = load_prompt_library()
category_choices = [(id, name) for id, name in prompt_library.get_categories()]

with gr.Blocks() as video_to_audio_tab:
    gr.Markdown("## MMAudio — Video-to-Audio Synthesis")
    gr.Markdown("Generate synchronized audio for videos with advanced configuration options.")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Input Video")
            
            with gr.Row():
                prompt_category = gr.Dropdown(
                    choices=[{"label": name, "value": id} for id, name in category_choices],
                    label="Sound Category",
                    info="Select the type of sound"
                )
                prompt_subcategory = gr.Dropdown(
                    choices=None,
                    label="Subcategory",
                    info="Select specific sound type"
                )
            
            prompt_select = gr.Dropdown(
                choices=None,
                label="Select Preset Prompt",
                info="Choose from preset prompts"
            )
            custom_prompt = gr.Textbox(
                label="Custom Prompt (Optional)",
                info="Enter your own prompt to override preset"
            )
            custom_negative = gr.Textbox(
                label="Custom Negative Prompt (Optional)",
                info="Enter custom negative prompt"
            )
            
        with gr.Column(scale=1):
            model_select = gr.Dropdown(choices=model_choices, value='large_44k_v2', label="Model Variant")
            quality = gr.Radio(["normal", "high"], value="normal", label="Quality Mode")
            num_steps = gr.Slider(1, 100, 25, label="Generation Steps", step=1)
            cfg_strength = gr.Slider(1, 10, 4.5, label="Guidance Strength", step=0.1)
            duration = gr.Slider(1, 30, 8, label="Duration (seconds)", step=0.5)
            seed = gr.Number(0, label="Random Seed", precision=0)
            
            with gr.Accordion("Advanced Options", open=False):
                full_precision = gr.Checkbox(label="Full Precision", value=False)
                mask_away_clip = gr.Checkbox(label="Mask Away CLIP", value=False)
                skip_video = gr.Checkbox(label="Output Audio Only", value=False)
    
    # Add submit button
    submit_btn = gr.Button("Generate")
    output_video = gr.Video(label="Generated Result")
    
    # Wire up the prompt selection updates
    prompt_category.change(
        fn=update_subcategories,
        inputs=[prompt_category],
        outputs=[prompt_subcategory]
    ).then(
        fn=update_prompts,
        inputs=[prompt_category, prompt_subcategory],
        outputs=[prompt_select]
    )
    
    prompt_subcategory.change(
        fn=update_prompts,
        inputs=[prompt_category, prompt_subcategory],
        outputs=[prompt_select]
    )
    
    # Submit button click event
    submit_btn.click(
        fn=video_to_audio,
        inputs=[
            video_input, prompt_category, prompt_subcategory, prompt_select,
            custom_prompt, custom_negative, model_select, seed, num_steps,
            cfg_strength, duration, mask_away_clip, full_precision, skip_video,
            quality
        ],
        outputs=output_video
    )

# Create initial choices for dropdown
prompt_library = load_prompt_library()
category_choices = [(id, name) for id, name in prompt_library.get_categories()]

# Create text-to-audio interface with the same dropdown setup
with gr.Blocks() as text_to_audio_tab:
    gr.Markdown("## MMAudio — Text-to-Audio Synthesis")
    gr.Markdown("Generate audio from text descriptions with advanced configuration options.")
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                prompt_category_text = gr.Dropdown(
                    choices=[{"label": name, "value": id} for id, name in category_choices],
                    label="Sound Category"
                )
                prompt_subcategory_text = gr.Dropdown(
                    choices=None,
                    label="Subcategory"
                )
            
            prompt_select_text = gr.Dropdown(
                choices=None,
                label="Select Preset Prompt"
            )
            custom_prompt_text = gr.Textbox(label="Custom Prompt (Optional)")
            custom_negative_text = gr.Textbox(label="Custom Negative Prompt (Optional)")
            
        with gr.Column(scale=1):
            model_select_text = gr.Dropdown(choices=model_choices, value='large_44k_v2', label="Model Variant")
            quality_text = gr.Radio(["normal", "high"], value="normal", label="Quality Mode")
            num_steps_text = gr.Slider(1, 100, 25, label="Generation Steps", step=1)
            cfg_strength_text = gr.Slider(1, 10, 4.5, label="Guidance Strength", step=0.1)
            duration_text = gr.Slider(1, 30, 8, label="Duration (seconds)", step=0.5)
            seed_text = gr.Number(0, label="Random Seed", precision=0)
            
            with gr.Accordion("Advanced Options", open=False):
                full_precision_text = gr.Checkbox(label="Full Precision", value=False)
    
    output_audio = gr.Audio(label="Generated Audio")
    
    # Add submit button
    submit_btn_text = gr.Button("Generate")
    
    # Wire up the prompt selection updates for text-to-audio
    prompt_category_text.change(
        fn=update_subcategories,
        inputs=[prompt_category_text],
        outputs=[prompt_subcategory_text]
    ).then(
        fn=update_prompts,
        inputs=[prompt_category_text, prompt_subcategory_text],
        outputs=[prompt_select_text]
    )
    
    prompt_subcategory_text.change(
        fn=update_prompts,
        inputs=[prompt_category_text, prompt_subcategory_text],
        outputs=[prompt_select_text]
    )
    
    # Submit button click event
    submit_btn_text.click(
        fn=text_to_audio,
        inputs=[
            prompt_category_text, prompt_subcategory_text, prompt_select_text,
            custom_prompt_text, custom_negative_text, model_select_text,
            seed_text, num_steps_text, cfg_strength_text, duration_text,
            full_precision_text, quality_text
        ],
        outputs=output_audio
    )

# Create the tabbed interface
demo = gr.TabbedInterface(
    [video_to_audio_tab, text_to_audio_tab],
    ['Video-to-Audio', 'Text-to-Audio'],
    title="MMAudio Synthesis"
)

if __name__ == "__main__":
    demo.launch(server_port=17888, allowed_paths=[output_dir], share=False)
