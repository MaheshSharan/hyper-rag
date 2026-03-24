from neo4j import GraphDatabase
from src.config.settings import settings
from typing import List, Dict, Any

class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def retrieve(self, user_query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """Graph expansion + centrality boost - fixed parameter conflict"""
        try:
            with self.driver.session() as session:
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
                return results

        except Exception as e:
            print(f"❌ Graph retriever error: {e}")
            return []