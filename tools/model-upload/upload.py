import os
import argparse

from huggingface_hub import HfApi, create_repo


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str)
parser.add_argument("--repo-name", type=str)
parser.add_argument("--branch-name", type=str, default="main")
args = parser.parse_args()

converted_ckpt: str = args.ckpt_path
repo_name: str = args.repo_name
branch_name: str = args.branch_name
try:
    create_repo(repo_name, repo_type="model", private=True)
except Exception as e:
    print(f"repo {repo_name} already exists! error: {e}")
    pass

files_and_dirs = os.listdir(converted_ckpt)
files = [f for f in files_and_dirs if os.path.isfile(os.path.join(converted_ckpt, f))]
print(f"Files to upload: {files}")

api = HfApi()
if branch_name != "main":
    try:
        api.create_branch(
            repo_id=repo_name,
            repo_type="model",
            branch=branch_name,
        )
    except Exception:
        print(f"branch {branch_name} already exists, try again...")

print(f"to upload: {files}")
for file in files:
    file_path = os.path.join(converted_ckpt, file)
    if os.path.isfile(file_path): # check if it's a file
        print(f"Uploading {file} to branch {branch_name}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file,
                repo_id=repo_name,
                repo_type="model",
                commit_message=f"Upload {file}",
                revision=branch_name,
            )
            print(f"Successfully uploaded {file} !")
        except Exception as e:
            print(f"Failed to upload {file}: {e}")