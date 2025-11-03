# import os
# import subprocess
# import sys
# from huggingface_hub import HfApi, HfFolder, upload_file

# # --- Run your training script ---
# print("\n 🚀 Running training script...")
# exit_code = subprocess.call([sys.executable, "-m", "src.model.main_loading_by_chunks"])
# if exit_code != 0:
#     print(" ❌ main_loading_by_chunks.py failed, aborting push.")
#     sys.exit(exit_code)

# # --- Push logs to GitHub (without model) ---
# try:
#     print("\n 📦 Adding code & logs only (no .pth files)...")
#     subprocess.run(["git", "add", "*.py"], check=False)
#     subprocess.run(["git", "add", "outputs/training_evaluation/"], check=False)

#     result = subprocess.run(["git", "diff", "--cached", "--quiet"])
#     if result.returncode != 0:
#         subprocess.run(["git", "commit", "-m", "Update training logs and code"], check=True)
#         subprocess.run(["git", "pull", "--rebase"], check=True)
#         subprocess.run(["git", "push", "origin", "main"], check=True)
#         print("\n ✅ Code + logs pushed to GitHub")
#     else:
#         print("No code/log changes to commit, skipping GitHub push.")

# except subprocess.CalledProcessError as e:
#     print(f"\n ❌ GitHub push failed: {e}")

import os
import subprocess
import sys
from huggingface_hub import HfApi, HfFolder, upload_file

# --- Run your training script ---
print("\n 🚀 Running training script...")
exit_code = subprocess.call([sys.executable, "-m", "src.model.main_loading_by_chunks"])
if exit_code != 0:
    print(" ❌ main_loading_by_chunks.py failed, aborting push.")
    sys.exit(exit_code)

# --- DEBUG: Check what was created ---
print("\n🔍 DEBUG: Checking created files...")
if os.path.exists("outputs/training_evaluation/"):
    print("Contents of outputs/training_evaluation/:")
    subprocess.run(["ls", "-la", "outputs/training_evaluation/"], check=False)
    
    # List all image files
    print("\nImage files found:")
    subprocess.run(["find", "outputs/training_evaluation/", "-type", "f", "(", "-name", "*.png", "-o", "-name", "*.jpg", "-o", "-name", "*.jpeg", ")"], check=False)

# --- Push logs and plots to GitHub ---
try:
    print("\n 📦 Adding code, logs, AND plots...")
    
    # Add specific file types to ensure plots are included
    subprocess.run(["git", "add", "*.py"], check=False)
    subprocess.run(["git", "add", "outputs/training_evaluation/*.png"], check=False)  # Add PNG plots
    subprocess.run(["git", "add", "outputs/training_evaluation/*.jpg"], check=False)  # Add JPG plots  
    subprocess.run(["git", "add", "outputs/training_evaluation/*.txt"], check=False)  # Add log files
    subprocess.run(["git", "add", "outputs/training_evaluation/*.json"], check=False) # Add JSON files
    subprocess.run(["git", "add", "outputs/training_evaluation/"], check=False)  # Catch anything else

    print("\nGit status after add:")
    subprocess.run(["git", "status"], check=False)

    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "commit", "-m", "Update training logs, code, and plots"], check=True)
        subprocess.run(["git", "pull", "--rebase"], check=True)
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
