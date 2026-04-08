from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id="medysaly/company-policy-rag",
    folder_path=".",
    repo_type="space",
    ignore_patterns=[
        ".venv/*",
        ".git/*",
        ".github/*",
        ".env",
        "__pycache__/*",
        ".pytest_cache/*",
        ".ruff_cache/*",
        "*.pyc",
        "tests/*",
        "scripts/*",
        "docs/*",
        "ui/*",
        "docker-compose.yml",
        "Makefile",
        "qdrant_data/*",
    ],
)
print("Deployed! Visit: https://huggingface.co/spaces/medysaly/company-policy-rag")
