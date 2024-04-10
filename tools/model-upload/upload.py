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

files = os.listdir(converted_ckpt)
print(f"files: {files}")

def upload_directory(directory_path, repo_name, branch_name, repo_type="model"):
    api = HfApi()

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            path_in_repo = os.path.relpath(file_path, start=directory_path)
            print(f"Uploading {file_path} to {path_in_repo}...")

            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_name,
                repo_type=repo_type,
                commit_message=f"Upload {path_in_repo}",
                revision=branch_name,
            )
            print(f"Successfully uploaded {path_in_repo} !")


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
    print(f"Uploading {file} to branch {branch_name}...")
    api.upload_file(
        path_or_fileobj=os.path.join(converted_ckpt, file),
        path_in_repo=file,
        repo_id=repo_name,
        repo_type="model",
        commit_message=f"Upload {file}",
        revision=branch_name,
    )
    print(f"Successfully uploaded {file} !")
