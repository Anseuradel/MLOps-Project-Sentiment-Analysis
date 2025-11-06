import os
import subprocess
import sys
from huggingface_hub import HfApi, HfFolder, upload_file

# --- Run training script ---
print("\n 🚀 Running training script...")
exit_code = subprocess.call([sys.executable, "-m", "src.model.main"])
if exit_code != 0:
    print(" ❌ main.py failed, aborting push.")
    sys.exit(exit_code)

# --- Push logs and plots to GitHub ---
try:
    print("\n 📦 Adding code, logs, AND plots...")
    
    # Add only the important files (skip cache files)
    subprocess.run(["git", "add", "*.py"], check=False)
    subprocess.run(["git", "add", "outputs/training_evaluation/"], check=False)
    subprocess.run(["git", "add", ".gitignore"], check=False)  # Add gitignore if updated

    print("\nGit status after add:")
    subprocess.run(["git", "status"], check=False)

    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "commit", "-m", "Update training logs, code, and plots"], check=True)
        
        # Simple pull without rebase (safer for this case)
        print("\n🔄 Pulling latest changes...")
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("\n ✅ Code + logs + plots pushed to GitHub")
    else:
        print("No changes to commit, skipping GitHub push.")

except subprocess.CalledProcessError as e:
    print(f"\n ❌ GitHub push failed: {e}")

# --- Push model to Hugging Face Hub ---
try:
    print("\n ☁️ Uploading model to Hugging Face Hub...")
    hf_repo = "Adelanseur/MLOps-Project"  # Hugging face repo
    token = HfFolder.get_token()

    model_path = "outputs/best_model.pth"
    if os.path.exists(model_path):
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.pth",
            repo_id=hf_repo,
            repo_type="model",
            token=token
        )
        print("✅ Model pushed to Hugging Face Hub")
    else:
        print("⚠️ No model file found at outputs/best_model.pth")

except Exception as e:
    print(f"❌ Hugging Face upload failed: {e}")
