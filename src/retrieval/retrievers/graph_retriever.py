import logging
from src.core.connection_pool import connection_pool
from typing import List, Dict, Any

logger = logging.getLogger("hyperrag.graph_retriever")


class GraphRetriever:
    def __init__(self):
        pass  # Using connection pool, no need to store driver

    def retrieve(self, user_query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """Graph expansion + centrality boost"""
        try:
            with connection_pool.neo4j.session() as session:
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.text CONTAINS $q OR ANY(k IN $keywords WHERE c.text CONTAINS k)
                    WITH c, count(*) as seed_score
                    OPTIONAL MATCH (c)-[r:IMPORTS|CALLS|CONTAINS]->(neighbor)
                    WITH c, collect(neighbor) as neighbors, seed_score
                    RETURN c.chunk_id as chunk_id,
                           c.text as text,
                           c.source as source,
                           size(neighbors) * 0.3 + seed_score as graph_score,
                           c.language as language
                    ORDER BY graph_score DESC
                    LIMIT $limit
                """, q=user_query[:200], keywords=user_query.split(), limit=top_k)

                results = []
                for record in result:
                    results.append({
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "source": record["source"],
                        "graph_score": float(record["graph_score"]),
                        "metadata": {"retriever": "graph"}
                    })
                
                logger.debug(f"Graph retriever: {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"Graph retriever error: {e}")
            return []