import ssl
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logging import logger
from app.models.base import Base

# Log the DB host at import time so Render logs confirm the URL being used.
# Password is masked — only host:port/dbname is shown.
try:
    from urllib.parse import urlparse as _urlparse

    _parsed = _urlparse(settings.database_url)
    logger.info(
        "DB engine target: %s:%s%s (driver: %s)",
        _parsed.hostname,
        _parsed.port,
        _parsed.path,
        _parsed.scheme,
    )
except Exception:
    pass  # Never block startup for a diagnostic log

# Supabase's Supavisor pooler uses a self-signed CA in its certificate chain.
# create_default_context() (CERT_REQUIRED) rejects it — we must disable
# verification. The connection is still TLS-encrypted; only cert-chain
# validation is skipped, which is the standard approach for Supabase poolers.
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=settings.db_pool_pre_ping,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    connect_args={
        "server_settings": {"application_name": "palmmind"},
        # Must be an ssl.SSLContext for asyncpg — string values like 'require'
        # are psycopg2 syntax and silently fail or raise in asyncpg.
        "ssl": _ssl_ctx,
        # Disables prepared-statement cache — required for Supabase's pooler
        # (port 6543) and harmless on the direct connection (port 5432).
        "statement_cache_size": 0,
        "timeout": 10,
    },
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        logger.info("Initializing database tables...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized.")
