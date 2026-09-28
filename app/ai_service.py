import json
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from app.config import settings, setup_logger

logger = setup_logger("ai_service")
client = AsyncOpenAI(api_key=settings.openai_api_key)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
async def get_embedding(text: str) -> list[float]:
    """
    Converts text into an embedding vector of size 1536.
    Used for both document ingestion and user queries.
    """
    try:
        text = text.replace("\n", " ")
        response = await client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error fetching embedding from OpenAI: {e}")
        raise

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
async def get_chat_response(query: str, context: str) -> dict:
    """
    Sends the user query and the retrieved context to the LLM to generate an answer.
    Returns a dictionary containing the 'answer' and a list of 'used_source_ids'.
    """
    system_prompt = (
        "You are an expert AI knowledge base assistant.\n"
        "Your task is to answer the user's question based strictly on the provided <context>.\n\n"
        "RULES:\n"
        "1. Answer ONLY using the information provided in the context.\n"
        "2. If the answer is not contained in the context, strictly output: 'I don't have information about this in my knowledge base.'\n"
        "3. You must output your response in valid JSON format with exactly two keys: 'answer' (string) and 'used_source_ids' (list of integers).\n"
        "4. In 'used_source_ids', list the IDs of the sources from the context that you used to construct your answer. If you couldn't answer, return an empty list [].\n"
        "5. Keep the answer concise and professional, using markdown formatting if appropriate (e.g., lists or bold text).\n\n"
        "<context>\n"
        f"{context}\n"
        "</context>"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error getting chat response from OpenAI: {e}")
        raise
