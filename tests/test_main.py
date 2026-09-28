import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import get_db

async def override_get_db():
    # Return an empty AsyncMock instead of the real database connection
    yield AsyncMock()

app.dependency_overrides[get_db] = override_get_db

@pytest.mark.asyncio
@patch('app.routers.rag.search_similar_chunks', new_callable=AsyncMock)
@patch('app.routers.rag.get_chat_response', new_callable=AsyncMock)
@patch('app.routers.rag.get_embedding', new_callable=AsyncMock)
@patch('app.repository.DocumentRepository.check_semantic_cache', new_callable=AsyncMock)
@patch('app.repository.DocumentRepository.save_to_semantic_cache', new_callable=AsyncMock)
async def test_ask_empty_database(mock_save_cache, mock_check_cache, mock_embed, mock_chat, mock_search):
    # Mock an empty search result from the database
    mock_search.return_value = []
    # Mock the LLM response
    mock_chat.return_value = {"answer": "No relevant documents found", "used_source_ids": []}
    # Mock the embedding generation
    mock_embed.return_value = [0.1] * 1536
    # Mock an empty semantic cache (Cache Miss)
    mock_check_cache.return_value = None
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/ask", json={"question": "What is the meaning of life?"})
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "No relevant documents found" in data["answer"]

@pytest.mark.asyncio
@patch('app.repository.DocumentRepository.get_all_document_names', new_callable=AsyncMock)
async def test_seed_missing_files(mock_get_names):
    mock_get_names.return_value = []
    
    with patch('glob.glob', return_value=[]):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/seed")
            
        assert response.status_code == 200
        assert "message" in response.json()
