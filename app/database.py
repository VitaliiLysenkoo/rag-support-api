from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings, setup_logger

logger = setup_logger("database")

# Replace postgresql:// with postgresql+asyncpg://
DATABASE_URL = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

try:
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    logger.info("Async database engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize async database engine: {e}")
    raise
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
