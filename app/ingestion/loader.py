from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(file_path: str) -> list:
    """Load a single document and return a list of pages/sections."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()


def load_directory(dir_path: str) -> list:
    """Load all supported documents from a directory."""
    path = Path(dir_path)

    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    documents = []
    for file_path in path.iterdir():
        if file_path.suffix.lower() in (".pdf", ".txt", ".md"):
            documents.extend(load_document(str(file_path)))

    return documents
