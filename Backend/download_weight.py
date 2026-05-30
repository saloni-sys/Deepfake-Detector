from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="abraraltaf92/deepfake-detection-models",
    filename="efficientnet_b4_best.pth",
    local_dir="weights"
)

print("Download complete!")