import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from pathlib import Path
from classes.document_ingestor import DocumentIngestor
from unittest.mock import MagicMock
import tempfile
import shutil

# Fixture to simulate tokenizer behavior
@pytest.fixture
def mock_tokenizer():
    mock = MagicMock()
    mock.tokenize.side_effect = lambda x: x.split()
    mock.convert_tokens_to_string.side_effect = lambda tokens: " ".join(tokens)
    mock.model_max_length = 512
    return mock

# Fixture to create a temporary input/output environment
@pytest.fixture
def temp_dirs():
    temp_input = tempfile.mkdtemp()
    temp_output = tempfile.mkdtemp()
    yield temp_input, temp_output
    shutil.rmtree(temp_input)
    shutil.rmtree(temp_output)

def test_txt_file_extraction_and_cleaning(temp_dirs, mock_tokenizer):
    input_dir, output_dir = temp_dirs
    file_name = "test.txt"
    test_path = Path(input_dir) / file_name
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("This is\na sample -\ntext with h t t p : / / example . com")

    ingestor = DocumentIngestor(
        file_list=[file_name],
        input_dir=input_dir,
        output_dir=output_dir,
        embedding_model_name="mock-model"
    )
    ingestor.tokenizer = mock_tokenizer  # inject mock tokenizer

    ingestor.process_files()

    # Check that chunks were saved
    output_files = list(Path(output_dir).glob("test_chunk_*.txt"))
    assert len(output_files) >= 1

    # Check the content of a chunk
    content = output_files[0].read_text(encoding="utf-8")
    assert "http://example.com" in content or "example.com" in content
    assert "-" not in content

def test_preprocess_text_cleans_url_and_formatting(mock_tokenizer):
    ingestor = DocumentIngestor([], ".", ".", "mock-model")
    ingestor.tokenizer = mock_tokenizer

    raw_text = "visit h t t p s : / / e x a m p l e . c o m \n\nLine with -\nbreak"
    preprocessed = ingestor._preprocess_text(raw_text)
    
    assert "https://example.com" in preprocessed or "example.com" in preprocessed
    assert "Line with break" in preprocessed
    assert "\n\n" not in preprocessed

def test_chunking_logic(mock_tokenizer):
    ingestor = DocumentIngestor([], ".", ".", "mock-model")
    ingestor.tokenizer = mock_tokenizer

    # Generate a long string
    long_text = "word " * 1000
    chunks = ingestor._chunk_text(long_text.strip(), max_tokens=100, overlap=10)
    
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 100 for chunk in chunks)

def test_extract_text_from_txt(temp_dirs, mock_tokenizer):
    input_dir, _ = temp_dirs
    txt_file = Path(input_dir) / "doc.txt"
    txt_file.write_text("Test content from txt file.", encoding="utf-8")

    ingestor = DocumentIngestor(["doc.txt"], input_dir, ".", "mock-model")
    ingestor.tokenizer = mock_tokenizer
    text = ingestor._extract_text_from_txt(txt_file)

    assert text == "Test content from txt file."
