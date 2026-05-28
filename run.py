# setup modal image
import modal
app = modal.App("nanogpt-from-scratch")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_file("transformer_notebook.py", "/root/transformer_notebook.py")
    .add_local_file("data/faust_combined.txt", "/root/data/faust_combined.txt")
)

@app.function(gpu="A100", image=image, timeout=1800)
def train():
    import subprocess, os
    os.chdir("/root")
    subprocess.run(["python", "-u", "/root/transformer_notebook.py"], check=True)

    # return the checkpoint bytes back to your laptop
    with open("/root/checkpoint_step3000.pt", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
     checkpoint_bytes = train.remote()
     with open("checkpoint_step3000.pt", "wb") as f:
        f.write(checkpoint_bytes)
     print("checkpoint saved locally!")   