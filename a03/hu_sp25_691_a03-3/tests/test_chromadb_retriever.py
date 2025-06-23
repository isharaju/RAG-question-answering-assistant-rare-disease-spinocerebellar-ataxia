import pytest
from unittest.mock import MagicMock, patch
from chromadb_retriever import ChromaDBRetriever

@pytest.fixture
def mock_retriever():
    with patch("chromadb_retriever.chromadb.PersistentClient") as MockClient, \
         patch("chromadb_retriever.SentenceTransformer") as MockEmbedder:

        # Mock client and collection
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        MockClient.return_value = mock_client_instance

        # Mock embedding model
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = [0.1, 0.2, 0.3]
        MockEmbedder.return_value = mock_embedding_model

        retriever = ChromaDBRetriever(
            embedding_model_name="mock-model",
            collection_name="test_collection",
            vectordb_dir="/tmp/mock_vectordb",
            score_threshold=0.2
        )
        return retriever, mock_collection

def test_embed_text(mock_retriever):
    retriever, _ = mock_retriever
    embedding = retriever.embed_text("sample text")
    assert isinstance(embedding, list)
    assert embedding == [0.1, 0.2, 0.3]

def test_extract_context_found(mock_retriever):
    retriever, _ = mock_retriever
    full_text = "Paragraph one.\n\nThis is a matching paragraph with Search term.\n\nAnother para."
    context = retriever.extract_context(full_text, "search term")
    assert "matching paragraph" in context.lower()

def test_extract_context_not_found(mock_retriever):
    retriever, _ = mock_retriever
    full_text = "This is just a random paragraph.\n\nNothing matches here."
    context = retriever.extract_context(full_text, "nonexistent term")
    assert context == full_text[:300]

def test_query_filters_and_returns_best(mock_retriever):
    retriever, mock_collection = mock_retriever

    # Mock collection query result
    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "metadatas": [[
            {"text": "This document contains the important search phrase.", "source": "paper1"},
            {"text": "Irrelevant document.", "source": "paper2"}
        ]],
        "distances": [[0.1, 0.9]]
    }

    results = retriever.query("search phrase", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["source"] == "paper1"
    assert results[0]["score"] == 0.1

def test_query_filters_by_score_threshold(mock_retriever):
    retriever, mock_collection = mock_retriever

    mock_collection.query.return_value = {
        "ids": [["doc1"]],
        "metadatas": [[{"text": "Not relevant text", "source": "source1"}]],
        "distances": [[0.05]]  # Below threshold of 0.2
    }

    results = retriever.query("unrelated query")
    assert results == []
