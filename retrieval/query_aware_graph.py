# -*- coding: utf-8 -*-
"""Query-aware graph retrieval: greedy best-first search over the citation graph.

At each step, expand the frontier paper with the highest semantic similarity
to the query. Uses CITES edges only in this base implementation. Non-CITES
edges (co-authorship, venue, field) are not utilised here -- see the separate
design notes for extension options.
"""
import heapq
import numpy as np


class MetaPathBestFirstGraph:
    """Query-aware greedy best-first over meta-paths.

    From each Paper, expand into candidates reachable via any of the four
    meta-paths:
      - CITES                         (Paper -[CITES]- Paper)
      - co-authorship                 (Paper -[WROTE]- Author -[WROTE]- Paper)
      - same venue                    (Paper -[PUBLISHED_IN]- Venue -[PUBLISHED_IN]- Paper)
      - same field of study           (Paper -[HAS_FIELD]- Field -[HAS_FIELD]- Paper)

    All meta-paths contribute candidates to a single Papers-only frontier,
    scored purely by cosine similarity to the query (equal weights across
    meta-path types). The frontier is a max-heap keyed on similarity; each
    pop yields the globally most-similar unexplored Paper.
    """

    CANDIDATE_QUERY = """
    MATCH (p:Paper {id: $id})-[:CITES]-(m:Paper) RETURN DISTINCT m.id AS id
    UNION
    MATCH (p:Paper {id: $id})-[:WROTE]-(:Author)-[:WROTE]-(m:Paper)
      WHERE m.id <> $id RETURN DISTINCT m.id AS id
    UNION
    MATCH (p:Paper {id: $id})-[:PUBLISHED_IN]-(:Venue)-[:PUBLISHED_IN]-(m:Paper)
      WHERE m.id <> $id RETURN DISTINCT m.id AS id
    UNION
    MATCH (p:Paper {id: $id})-[:HAS_FIELD]-(:FieldOfStudy)-[:HAS_FIELD]-(m:Paper)
      WHERE m.id <> $id RETURN DISTINCT m.id AS id
    """

    def __init__(self, neo4j_driver, embed_model, paper_embeddings):
        """
        Args:
            neo4j_driver:      Neo4j driver
            embed_model:       SentenceTransformer (used for query embedding)
            paper_embeddings:  dict paper_id -> np.array (precomputed once)
        """
        self.driver = neo4j_driver
        self.embed_model = embed_model
        self.paper_embeddings = paper_embeddings

    def _paper_embedding(self, paper_id):
        return self.paper_embeddings.get(paper_id)

    @staticmethod
    def _cosine(a, b):
        if a is None or b is None:
            return -1.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _get_meta_neighbors(self, session, paper_id):
        return [r["id"] for r in session.run(self.CANDIDATE_QUERY, id=paper_id)]

    def retrieve(self, query, seed_paper_id, k, max_hops=4, sim_threshold=0.0):
        query_vec = self.embed_model.encode([query])[0]
        frontier = []
        counter = 0
        visited = {seed_paper_id}
        found = []

        with self.driver.session() as session:
            for nid in self._get_meta_neighbors(session, seed_paper_id):
                if nid not in visited:
                    visited.add(nid)
                    sim = self._cosine(query_vec, self._paper_embedding(nid))
                    heapq.heappush(frontier, (-sim, counter, nid, 1))
                    counter += 1

            while frontier and len(found) < k:
                neg_sim, _, pid, depth = heapq.heappop(frontier)
                if -neg_sim < sim_threshold:   # heap is max-by-sim; all remaining are worse
                    break
                found.append({
                    "paper_id": pid,
                    "distance": depth,
                    "similarity": -neg_sim,
                })
                if depth >= max_hops:
                    continue
                for nid in self._get_meta_neighbors(session, pid):
                    if nid not in visited:
                        visited.add(nid)
                        sim = self._cosine(query_vec, self._paper_embedding(nid))
                        heapq.heappush(frontier, (-sim, counter, nid, depth + 1))
                        counter += 1

        return found


class GreedyBestFirstGraph:
    def __init__(self, neo4j_driver, embed_model, paper_texts):
        """
        Args:
            neo4j_driver:  Neo4j driver
            embed_model:   SentenceTransformer (same one used for vector search)
            paper_texts:   dict paper_id -> text (title + abstract) for on-the-fly
                           paper embedding
        """
        self.driver = neo4j_driver
        self.embed_model = embed_model
        self.paper_texts = paper_texts

    def _paper_embedding(self, paper_id):
        text = self.paper_texts.get(paper_id, "") or ""
        if not text.strip():
            return None
        return self.embed_model.encode([text])[0]

    @staticmethod
    def _cosine(a, b):
        if a is None or b is None:
            return -1.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _get_cites_neighbors(self, session, paper_id):
        q = "MATCH (p:Paper {id: $id})-[:CITES]-(m:Paper) RETURN m.id AS id"
        return [r["id"] for r in session.run(q, id=paper_id)]

    def retrieve(self, query, seed_paper_id, k, max_hops=4):
        """Greedy best-first search from a seed paper, guided by query similarity."""
        query_vec = self.embed_model.encode([query])[0]

        # Min-heap of (-similarity, tie_breaker, paper_id, depth)
        # tie_breaker is an insertion counter to avoid dict-like comparison
        frontier = []
        counter = 0
        visited = {seed_paper_id}
        found = []

        with self.driver.session() as session:
            # Seed the frontier with the seed paper's direct neighbors
            for nid in self._get_cites_neighbors(session, seed_paper_id):
                if nid not in visited:
                    visited.add(nid)
                    sim = self._cosine(query_vec, self._paper_embedding(nid))
                    heapq.heappush(frontier, (-sim, counter, nid, 1))
                    counter += 1

            while frontier and len(found) < k:
                neg_sim, _, pid, depth = heapq.heappop(frontier)
                found.append({
                    "paper_id": pid,
                    "distance": depth,
                    "similarity": -neg_sim,
                })
                if depth >= max_hops:
                    continue
                for nid in self._get_cites_neighbors(session, pid):
                    if nid not in visited:
                        visited.add(nid)
                        sim = self._cosine(query_vec, self._paper_embedding(nid))
                        heapq.heappush(frontier, (-sim, counter, nid, depth + 1))
                        counter += 1

        return found
