import os
import subprocess
import sys

# --- Run your training script ---
print("\n Running training script...")
exit_code = subprocess.call([sys.executable, "-m", "src.model.main"])
if exit_code != 0:
    print(" main.py failed, aborting push.")
    sys.exit(exit_code)

# --- Git commands for Colab auto-push ---
try:
    print("\n Adding changes first...")
    subprocess.run(["git", "add", "outputs/"], check=True)
    # subprocess.run(["git", "add", "chunks.txt"], check=False)  # may not exist yet

    # Auto-save commit (only if changes exist)
    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "commit", "-m", "Auto-save before pull"], check=True)
    else:
        print("No staged changes to commit before pull.")

    # Force add all changes
    subprocess.run(["git", "add", "-A"], check=True)
    
    # Commit only if there are staged changes
    subprocess.run(["git", "commit", "-m", "Auto-save before pull"], check=False)


    print("\n Pulling latest changes...")
    subprocess.run(["git", "pull", "--rebase"], check=True)

    print("\n Adding new training outputs...")
    subprocess.run(["git", "add", "outputs/"], check=True)
    # subprocess.run(["git", "add", "chunks.txt"], check=False)

    # Final commit (only if changes exist)
    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "commit", "-m", "Update model & outputs from Colab"], check=True)
    else:
        print("No changes to commit, skipping final commit step.")

    print("\n Pushing to GitHub...")
    subprocess.run(["git", "push", "origin", "main"], check=True)

    print("\n ✅ Training results pushed successfully!")

except subprocess.CalledProcessError as e:
    print(f"\n ❌ Git operation failed: {e}")
    sys.exit(1)
