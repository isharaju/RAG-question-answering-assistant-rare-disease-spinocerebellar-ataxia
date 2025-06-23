import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

from embedding_preparer import EmbeddingPreparer

@pytest.fixture
def mock_embedding_preparer_env():
    with patch("embedding_preparer.AutoTokenizer") as MockTokenizer, \
         patch("embedding_preparer.AutoModel") as MockModel:

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.return_value = mock_output

        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        MockModel.from_pretrained.return_value = mock_model

        preparer = EmbeddingPreparer(
            file_list=["doc1.txt"],
            input_dir="/tmp/input",
            output_dir="/tmp/output",
            embedding_model_name="mock-model"
        )
        preparer.tokenizer = mock_tokenizer
        preparer.model = mock_model

        return preparer

def test_read_file(mock_embedding_preparer_env):
    preparer = mock_embedding_preparer_env
    test_path = Path("/tmp/doc1.txt")
    with patch("builtins.open", mock_open(read_data="sample text")):
        result = preparer._read_file(test_path)
        assert result == "sample text"

def test_generate_embedding(mock_embedding_preparer_env):
    preparer = mock_embedding_preparer_env

    # Patch tokenizer and model behavior
    preparer.tokenizer.__call__.return_value = {"input_ids": [[1, 2, 3]]}
    preparer.model.__call__.return_value.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    with patch("torch.no_grad"):
        result = preparer._generate_embedding("test text")
        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]

def test_save_embedding(mock_embedding_preparer_env):
    preparer = mock_embedding_preparer_env
    output_file = Path("/tmp/input/doc1.txt")
    embedding = [0.1, 0.2, 0.3]

    with patch("builtins.open", mock_open()) as m_open:
        preparer._save_embedding(output_file, embedding)
        handle = m_open()
        handle.write.assert_called()  # Check that something was written

def test_process_files_success(mock_embedding_preparer_env):
    preparer = mock_embedding_preparer_env
    input_file_path = Path("/tmp/input/doc1.txt")
    embedding = [0.1, 0.2, 0.3]

    with patch.object(Path, "exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="text")), \
         patch("torch.no_grad"), \
         patch("json.dump") as mock_json_dump:

        preparer._generate_embedding = MagicMock(return_value=embedding)
        preparer._save_embedding = MagicMock()
        preparer.process_files()

        preparer._generate_embedding.assert_called_once()
        preparer._save_embedding.assert_called_once()
