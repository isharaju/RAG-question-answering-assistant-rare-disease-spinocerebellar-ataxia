import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

from embedding_loader import EmbeddingLoader

@pytest.fixture
def mock_loader_env():
    with patch("embedding_loader.chromadb.PersistentClient") as MockClient:
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        MockClient.return_value = mock_client

        loader = EmbeddingLoader(
            cleaned_text_file_list=["doc1.txt"],
            cleaned_text_dir="/tmp/texts",
            embeddings_dir="/tmp/embeddings",
            vectordb_dir="/tmp/vectordb",
            collection_name="test_collection"
        )
        return loader, mock_collection

def test_load_cleaned_text_success(mock_loader_env):
    loader, _ = mock_loader_env
    test_path = Path("/fake/doc.txt")
    text_content = "Cleaned text content."

    with patch("builtins.open", mock_open(read_data=text_content)):
        result = loader._load_cleaned_text(test_path)
        assert result == text_content

def test_load_cleaned_text_failure(mock_loader_env):
    loader, _ = mock_loader_env
    test_path = Path("/nonexistent/doc.txt")

    with patch("builtins.open", side_effect=FileNotFoundError()):
        result = loader._load_cleaned_text(test_path)
        assert result == ""

def test_load_embeddings_success(mock_loader_env):
    loader, _ = mock_loader_env
    embedding_path = Path("/fake/embeddings.json")
    fake_embeddings = [0.1, 0.2, 0.3]

    with patch("builtins.open", mock_open(read_data=json.dumps(fake_embeddings))):
        result = loader._load_embeddings(embedding_path)
        assert result == fake_embeddings

def test_load_embeddings_invalid_format(mock_loader_env):
    loader, _ = mock_loader_env
    embedding_path = Path("/fake/embeddings.json")
    with patch("builtins.open", mock_open(read_data=json.dumps({"not": "a list"}))):
        result = loader._load_embeddings(embedding_path)
        assert isinstance(result, list)

def test_process_files_success(mock_loader_env):
    loader, mock_collection = mock_loader_env

    text_content = "Sample cleaned text."
    embedding_content = [0.4, 0.5, 0.6]

    cleaned_text_path = Path("/tmp/texts/doc1.txt")
    embedding_file_path = Path("/tmp/embeddings/doc1_embeddings.json")

    with patch.object(Path, "exists", return_value=True), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.load", return_value=embedding_content):

        m_open().read.return_value = text_content
        loader.process_files()

        mock_collection.add.assert_called_once()
        args = mock_collection.add.call_args[1]

        assert args["ids"] == ["doc1.txt"]
        assert args["embeddings"] == [embedding_content]
        assert args["metadatas"][0]["text"] == text_content
        assert args["metadatas"][0]["source"] == "doc1.txt"

def test_process_files_missing_files(mock_loader_env):
    loader, mock_collection = mock_loader_env

    with patch.object(Path, "exists", return_value=False):
        loader.process_files()

    mock_collection.add.assert_not_called()
