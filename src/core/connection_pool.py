"""
Connection Pool Manager - Singleton pattern for database connections
Prevents connection leaks and improves performance
"""
import logging
from typing import Optional
from qdrant_client import QdrantClient
from opensearchpy import OpenSearch
from neo4j import GraphDatabase, Driver
from src.config.settings import settings

logger = logging.getLogger("hyperrag.connections")


class ConnectionPool:
    """Singleton connection pool for all database clients"""
    _instance: Optional['ConnectionPool'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._qdrant_client: Optional[QdrantClient] = None
        self._opensearch_client: Optional[OpenSearch] = None
        self._neo4j_driver: Optional[Driver] = None
        self._initialized = True
        logger.info("ConnectionPool initialized")
    
    @property
    def qdrant(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                url=settings.QDRANT_HOST,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=60
            )
            logger.info("✅ Qdrant connection established")
        return self._qdrant_client
    
    @property
    def opensearch(self) -> OpenSearch:
        """Get or create OpenSearch client"""
        if self._opensearch_client is None:
            self._opensearch_client = OpenSearch(
                hosts=[settings.OPENSEARCH_HOST],
                http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD),
                verify_certs=False,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            logger.info("✅ OpenSearch connection established")
        return self._opensearch_client
    
    @property
    def neo4j(self) -> Driver:
        """Get or create Neo4j driver"""
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            logger.info("✅ Neo4j connection established")
        return self._neo4j_driver
    
    def close_all(self):
        """Close all connections - call on shutdown"""
        if self._qdrant_client:
            self._qdrant_client.close()
            logger.info("Qdrant connection closed")
        
        if self._opensearch_client:
            self._opensearch_client.close()
            logger.info("OpenSearch connection closed")
        
        if self._neo4j_driver:
            self._neo4j_driver.close()
            logger.info("Neo4j connection closed")
        
        self._qdrant_client = None
        self._opensearch_client = None
        self._neo4j_driver = None


# Global singleton instance
connection_pool = ConnectionPool()
