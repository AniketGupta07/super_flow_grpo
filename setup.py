from setuptools import setup, find_packages

setup(
    name="flow-grpo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio",
        # "transformers==4.40.0",
        "transformers==4.53.0",
        "accelerate==1.4.0",
        "diffusers==0.33.1", 
        
        # "flash-attn==2.7.4.post1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",
        
        "albumentations==1.4.10",  
        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        
        "tqdm==4.67.1",
        "wandb==0.18.7",
        "pydantic==2.10.6",  
        "requests==2.32.3",
        "matplotlib==3.10.0",
        
        "deepspeed==0.16.4",  
        "peft==0.10.0",       
        "bitsandbytes==0.45.3",
        
        "aiohttp==3.11.13",
        "fastapi==0.115.11", 
        "uvicorn==0.34.0",
        
        # "huggingface-hub==0.29.1",
        "huggingface-hub>=0.30.0,<1.0",  
        "datasets==3.3.2",
        # "tokenizers==0.19.1",
        "tokenizers==0.21.0",
        
        "einops==0.8.1",
        "nvidia-ml-py==12.570.86",
        "xformers",
        "absl-py",
        "ml_collections",
        "sentencepiece",
        "openai",
        "sentence-transformers",
        "qwen-vl-utils",
        # torchcodec declares no torch requirement, so an unpinned install resolves to a build
        # for a newer torch and fails to load its shared library. 0.2 is the torch 2.6 match.
        "torchcodec==0.2.1",
        # Video decoding backend qwen_vl_utils falls back to via torchvision.io.read_video,
        # used by the qwenvl/VPS/videoscore2 rewards.
        "av",
        "imageio",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg",
        "gradio>=5.0.0",
    ],
    # flash-attn's setup imports torch; pip's isolated build env has no torch.
    # Install: pip install -e ".[flash]" --no-build-isolation  (torch must be in env first)
    extras_require={
        "flash": ["flash-attn==2.7.4.post1"],
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
