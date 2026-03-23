# download_model.py
from huggingface_hub import snapshot_download
import os

# 设置国内镜像（解决下载慢/失败）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型名称和保存路径
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
local_dir = "./DeepSeek-R1-Distill-Qwen-7B"

# 开始下载（支持断点续传）
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4  # 多线程加速
)

print(f"模型已下载到：{os.path.abspath(local_dir)}")