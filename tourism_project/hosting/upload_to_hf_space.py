from huggingface_hub import HfApi
import os

api = HfApi()

space_repo_id = "Swetha1929/tourism-package-space"

# Create the public Hugging Face Space
api.create_repo(
    repo_id=space_repo_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

# Upload deployment files to the Space
api.upload_file(
    path_or_fileobj="tourism_project/deployment/app.py",
    path_in_repo="app.py",
    repo_id=space_repo_id,
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="tourism_project/deployment/requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=space_repo_id,
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="tourism_project/deployment/Dockerfile",
    path_in_repo="Dockerfile",
    repo_id=space_repo_id,
    repo_type="space"
)

print("Space repository created and deployment files uploaded successfully!")
print("Space Repository:", space_repo_id)
