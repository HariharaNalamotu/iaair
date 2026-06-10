#!/usr/bin/env python3
"""
Build the Neo4j knowledge graph from committed CSVs --- no API, no dump file.

This is the standalone import half of ingestion/run_ingestion.py. It reads the
committed edge/node CSVs in data/ and (re)creates the exact 10,199-node graph
used by every retrieval method, so the graph is reproducible from repo data
alone (no Semantic Scholar API key and no transfer/neo4j.dump required).

Reads:
    data/papers.csv          data/citations.csv
    data/authors.csv         data/written_by.csv
    data/venues.csv          data/write_together.csv
    data/fields_of_study.csv data/written_for.csv
    data/institutions.csv    data/field_of_study.csv
                             data/affiliations.csv

Connects using NEO4J_URI / NEO4J_USER / NEO4J_PASS from .env.

Run:
    python scripts/build_graph.py            # build (refuses if graph non-empty)
    python scripts/build_graph.py --force    # wipe existing graph, then build
"""
import argparse, csv, os, pathlib, sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # full_text fields are large
BATCH = 500


def read_rows(filename):
    path = DATA / filename
    if not path.exists():
        print(f"  [warn] {filename} missing — skipping")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_batched(session, query, data, batch=BATCH):
    for i in range(0, len(data), batch):
        session.run(query, batch=data[i:i + batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="wipe any existing graph before building")
    args = ap.parse_args()

    uri  = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    pw   = os.environ["NEO4J_PASS"]
    driver = GraphDatabase.driver(uri, auth=(user, pw))

    with driver.session() as session:
        n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if n and not args.force:
            driver.close()
            sys.exit(f"Graph already has {n} nodes. Re-run with --force to wipe "
                     "and rebuild.")
        if n:
            print(f"  Wiping existing graph ({n} nodes)…")
            session.run("MATCH (n) DETACH DELETE n")

        for label in ["Paper", "Author", "Venue", "FieldOfStudy", "Institution"]:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        print("  Constraints ready.")

        # ── Nodes ────────────────────────────────────────────────────────────
        papers = read_rows("papers.csv")
        run_batched(session, """
            UNWIND $batch AS p
            MERGE (n:Paper {id: p.id})
            SET n.title=p.title, n.abstract=p.abstract,
                n.year=toInteger(p.year),
                n.citationCount=toInteger(p.citationCount),
                n.referenceCount=toInteger(p.referenceCount),
                n.isOpenAccess=p.isOpenAccess, n.venue=p.venue,
                n.externalIds=p.externalIds, n.full_text=p.full_text
        """, papers)
        print(f"  Paper nodes: {len(papers)}")

        authors = read_rows("authors.csv")
        run_batched(session, """
            UNWIND $batch AS a MERGE (n:Author {id:a.authorId}) SET n.name=a.name
        """, authors)
        print(f"  Author rows: {len(authors)}")

        run_batched(session, "UNWIND $batch AS v MERGE (n:Venue {id:v.venue})",
                    read_rows("venues.csv"))
        run_batched(session, "UNWIND $batch AS f MERGE (n:FieldOfStudy {id:f.fieldOfStudy})",
                    read_rows("fields_of_study.csv"))
        run_batched(session, "UNWIND $batch AS i MERGE (n:Institution {id:i.institution})",
                    read_rows("institutions.csv"))
        print("  Venue / FieldOfStudy / Institution nodes created.")

        # ── Relationships ────────────────────────────────────────────────────
        run_batched(session, """
            UNWIND $batch AS r
            MATCH (a:Paper {id:r.cites}) MATCH (b:Paper {id:r.cited_by})
            MERGE (a)-[:CITES]->(b)
        """, read_rows("citations.csv"))

        run_batched(session, """
            UNWIND $batch AS r
            MERGE (a:Author {id:r.written_by}) MERGE (p:Paper {id:r.writes})
            MERGE (a)-[:WROTE]->(p)
        """, read_rows("written_by.csv"))

        run_batched(session, """
            UNWIND $batch AS r
            MERGE (a:Author {id:r.author1}) MERGE (b:Author {id:r.author2})
            MERGE (a)-[:WRITES_WITH]->(b)
        """, read_rows("write_together.csv"))

        run_batched(session, """
            UNWIND $batch AS r
            MERGE (p:Paper {id:r.paperId}) MERGE (v:Venue {id:r.venue})
            MERGE (p)-[:PUBLISHED_IN]->(v)
        """, read_rows("written_for.csv"))

        run_batched(session, """
            UNWIND $batch AS r
            MERGE (p:Paper {id:r.paperId}) MERGE (f:FieldOfStudy {id:r.fieldOfStudy})
            MERGE (p)-[:HAS_FIELD]->(f)
        """, read_rows("field_of_study.csv"))

        run_batched(session, """
            UNWIND $batch AS r
            MERGE (a:Author {id:r.authorId}) MERGE (i:Institution {id:r.affiliation})
            MERGE (a)-[:AFFILIATED_WITH]->(i)
        """, read_rows("affiliations.csv"))
        print("  Relationships created.")

        # ── Summary ──────────────────────────────────────────────────────────
        counts = session.run("""
            MATCH (n) WITH labels(n)[0] AS l, count(*) AS c
            RETURN l, c ORDER BY l
        """)
        print("\n  Node counts:")
        total = 0
        for rec in counts:
            print(f"    {rec['l']:<14} {rec['c']}")
            total += rec["c"]
        print(f"    {'TOTAL':<14} {total}")

    driver.close()
    print("\nGraph build complete.")


if __name__ == "__main__":
    main()
