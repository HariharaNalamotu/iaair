#!/usr/bin/env python3
"""
Full ingestion pipeline: fetches 1500 papers from Semantic Scholar via BFS,
embeds chunks into Milvus, exports CSVs, and imports everything into Neo4j.
Run from /Users/Hari/IAAIR/
"""
import requests, time, csv, json, os, sys, pathlib
from dotenv import load_dotenv
from itertools import combinations
from typing import List

import tiktoken
from pypdf import PdfReader
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
API_KEYS = [
    os.environ["SEMANTIC_SCHOLAR_API_KEY_1"],
    os.environ["SEMANTIC_SCHOLAR_API_KEY_2"],
]
_call_count = 0

PDF_DIR       = str(ROOT / "semantic_scholar_corpus" / "pdfs") + "/"
MILVUS_DB     = str(ROOT / "RAG.db")
COLLECTION    = "ingestion_v0"
VECTOR_DIM    = 768
STATE_FILE    = str(ROOT / "ingestion" / "ingestion_state.json")

N_PAPERS      = 1500
CHUNK_SIZE    = 128
CHUNK_OVERLAP = 26
EMBED_MODEL   = "jordyvl/scibert_scivocab_uncased_sentence_transformer"
SEED_PAPER_ID = "649def34f8be52c8b66281af98ae884c09aef38b"

FLUSH_EVERY    = 50
PDF_BATCH_SIZE = 500

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def progress_bar(n, total, desc="Progress", extra=""):
    pct = n / total if total else 0
    filled = int(45 * pct)
    bar = "█" * filled + "░" * (45 - filled)
    elapsed = time.time() - _t0
    rate = n / elapsed if elapsed else 0
    eta = int((total - n) / rate) if rate else 0
    print(f"\r{desc}  [{bar}]  {n}/{total} ({pct:.0%})  {rate:.2f}/s  ETA {eta}s  {extra[:60]}", end="", flush=True)

def getPaper(paperId=SEED_PAPER_ID):
    global _call_count
    url = f"http://api.semanticscholar.org/graph/v1/paper/{paperId}"
    params = {
        "fields": ("title,year,abstract,citationCount,referenceCount,"
                   "citations,authors,isOpenAccess,openAccessPdf,"
                   "venue,fieldsOfStudy,externalIds")
    }
    headers = {"x-api-key": API_KEYS[_call_count % 2]}
    _call_count += 1
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def download_pdf(url, file_name):
    save_path = PDF_DIR + file_name + ".pdf"
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False

def read_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() or "" for page in reader.pages)

def chunk_text_by_tokens(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        chunks.append(enc.decode(tokens[start : start + chunk_size]))
        start += chunk_size - overlap
    return chunks

DATA_DIR = ROOT / "data"

def append_csv(filename, rows):
    if not rows:
        return
    path = DATA_DIR / filename
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)
def create_embedding(texts):
    return embed_model.encode(texts)

print("Setting up Milvus...")
client = MilvusClient(MILVUS_DB)
if client.has_collection(collection_name=COLLECTION):
    count = client.query(collection_name=COLLECTION, filter="id >= 0",
                         output_fields=["count(*)"])[0]["count(*)"]
    print(f"  Collection exists with {count} vectors — will append.")
else:
    client.create_collection(collection_name=COLLECTION, dimension=VECTOR_DIM,
                             enable_dynamic_field=True)
    print(f"  Created new collection '{COLLECTION}'.")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD STATE
# ═══════════════════════════════════════════════════════════════════════════════
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        _state = json.load(f)
    seen_paper_ids = set(_state["seen_paper_ids"])
    paper_queue    = _state["paper_queue"]
    id_num         = _state["id_num"]
    print(f"Resuming: {len(seen_paper_ids)} seen, {len(paper_queue)} queued, id_num={id_num}")
else:
    seen_paper_ids = set()
    paper_queue    = []
    id_num         = 0
    print("Starting fresh ingestion.")

# Neo4j buffers
searched_papers = []
authors = []
venues = []
institutions = []
fields_of_study = []
relation_citations = []
relation_written_by = []
relation_write_together = []
relation_written_for = []
relation_field_of_study = []
relation_affiliations = []
abstract_datapoints = []

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INGESTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"Ingesting {N_PAPERS} papers...")
print(f"{'='*70}\n")

_t0 = time.time()
new_papers_count = 0
consecutive_skips = 0
MAX_SKIPS = 2000
errors = []

while new_papers_count < N_PAPERS:
    try:
        paper_id = paper_queue.pop(0)
    except IndexError:
        paper_id = SEED_PAPER_ID

    if paper_id in seen_paper_ids:
        consecutive_skips += 1
        if consecutive_skips >= MAX_SKIPS:
            print(f"\nStopped: {MAX_SKIPS} consecutive already-seen IDs.")
            break
        continue
    consecutive_skips = 0

    paper = getPaper(paper_id)
    if paper is None:
        errors.append(f"API None: {paper_id}")
        continue

    seen_paper_ids.add(paper_id)
    new_papers_count += 1

    for citation in paper.get("citations", []):
        if citation.get("paperId"):
            paper_queue.append(citation["paperId"])

    abstract = paper.get("abstract") or ""
    venue = paper.get("venue") or ""

    if venue and {"venue": venue} not in venues:
        venues.append({"venue": venue})
    if venue:
        relation_written_for.append({"paperId": paper["paperId"], "venue": venue})

    # PDF
    full_text = None
    open_pdf = paper.get("openAccessPdf") or {}
    pdf_url = open_pdf.get("url", "")
    if pdf_url:
        pdf_path = f"{PDF_DIR}{paper['paperId']}.pdf"
        ok = download_pdf(pdf_url, paper["paperId"])
        if ok:
            try:
                full_text = read_pdf_text(pdf_path)
            except Exception as e:
                errors.append(f"PDF read: {paper['paperId']}: {e}")
        else:
            errors.append(f"PDF dl fail: {paper['paperId']}")

    # Paper node
    searched_papers.append({
        "id":             paper["paperId"],
        "title":          paper["title"],
        "abstract":       abstract,
        "year":           paper.get("year"),
        "citationCount":  paper.get("citationCount"),
        "referenceCount": paper.get("referenceCount"),
        "isOpenAccess":   paper.get("isOpenAccess"),
        "venue":          venue,
        "externalIds":    str(paper.get("externalIds", {})),
        "full_text":      full_text,
    })

    # Authors
    paper_authors = paper.get("authors", [])
    for author in paper_authors:
        authors.append(author)
        relation_written_by.append({
            "written_by": author["authorId"],
            "writes":     paper["paperId"],
        })
    for pair in combinations(paper_authors, 2):
        relation_write_together.append({
            "author1": pair[0]["authorId"],
            "author2": pair[1]["authorId"],
        })

    # Citations
    for citation in paper.get("citations", []):
        relation_citations.append({
            "cites":    paper["paperId"],
            "cited_by": citation["paperId"],
        })

    # Fields of study
    for field in paper.get("fieldsOfStudy") or []:
        if {"fieldOfStudy": field} not in fields_of_study:
            fields_of_study.append({"fieldOfStudy": field})
        relation_field_of_study.append({
            "paperId":      paper["paperId"],
            "fieldOfStudy": field,
        })

    # Abstract chunks → Milvus
    if abstract:
        text_chunks = chunk_text_by_tokens(abstract)
        embeddings = create_embedding(text_chunks)
        for chunk_idx, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
            abstract_datapoints.append({
                "id":           id_num,
                "paperId":      paper["paperId"],
                "title":        paper["title"],
                "abstract":     abstract,
                "chunk_text":   chunk,
                "chunk_index":  chunk_idx,
                "chunk_source": "abstract",
                "vector":       emb,
            })
            id_num += 1

    # Periodic flush
    if new_papers_count % FLUSH_EVERY == 0:
        if abstract_datapoints:
            client.insert(collection_name=COLLECTION, data=abstract_datapoints)
            abstract_datapoints.clear()
        for fname, buf in [
            ("papers.csv",         searched_papers),
            ("authors.csv",        authors),
            ("venues.csv",         venues),
            ("citations.csv",      relation_citations),
            ("written_by.csv",     relation_written_by),
            ("write_together.csv", relation_write_together),
            ("written_for.csv",    relation_written_for),
            ("field_of_study.csv", relation_field_of_study),
        ]:
            append_csv(fname, buf)
            buf.clear()
        with open(STATE_FILE, "w") as f:
            json.dump({"seen_paper_ids": list(seen_paper_ids),
                       "paper_queue": paper_queue, "id_num": id_num}, f)
        print(f"\n  [flush @ {new_papers_count}] saved to disk")

    progress_bar(new_papers_count, N_PAPERS, "Papers", paper["title"] or "")
    time.sleep(0.5)

# Final flush
if abstract_datapoints:
    client.insert(collection_name=COLLECTION, data=abstract_datapoints)
    abstract_datapoints.clear()
for fname, buf in [
    ("papers.csv",         searched_papers),
    ("authors.csv",        authors),
    ("venues.csv",         venues),
    ("citations.csv",      relation_citations),
    ("written_by.csv",     relation_written_by),
    ("write_together.csv", relation_write_together),
    ("written_for.csv",    relation_written_for),
    ("field_of_study.csv", relation_field_of_study),
]:
    append_csv(fname, buf)
    buf.clear()

with open(STATE_FILE, "w") as f:
    json.dump({"seen_paper_ids": list(seen_paper_ids),
               "paper_queue": paper_queue, "id_num": id_num}, f)

elapsed = time.time() - _t0
print(f"\n\n{'='*70}")
print(f"Paper ingestion complete: {new_papers_count} papers in {elapsed:.0f}s")
print(f"  {len(seen_paper_ids)} total seen | {id_num} abstract vectors")
print(f"  Errors: {len(errors)}")
print(f"{'='*70}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# AUTHOR AFFILIATIONS BATCH
# ═══════════════════════════════════════════════════════════════════════════════
print("Fetching author affiliations...")
AUTHOR_API = "https://api.semanticscholar.org/graph/v1/author/batch"

# Read all unique author IDs from CSV
all_author_ids = set()
if (DATA_DIR / "authors.csv").exists():
    with open(DATA_DIR / "authors.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("authorId"):
                all_author_ids.add(row["authorId"])
author_ids = list(all_author_ids)
print(f"  {len(author_ids)} unique authors to look up")

BATCH = 500
for batch_start in range(0, len(author_ids), BATCH):
    batch = author_ids[batch_start : batch_start + BATCH]
    try:
        response = requests.post(
            AUTHOR_API,
            params={"fields": "authorId,name,affiliations"},
            json={"ids": batch},
            headers={"x-api-key": API_KEYS[0]},
            timeout=30,
        ).json()
        for author_data in response:
            if author_data is None:
                continue
            for affiliation in author_data.get("affiliations", []):
                relation_affiliations.append({
                    "authorId":    author_data["authorId"],
                    "affiliation": affiliation,
                })
                if {"institution": affiliation} not in institutions:
                    institutions.append({"institution": affiliation})
    except Exception as e:
        print(f"  Author batch error: {e}")
    print(f"  Processed {min(batch_start + BATCH, len(author_ids))}/{len(author_ids)} authors")
    time.sleep(1)

append_csv("institutions.csv", institutions)
append_csv("affiliations.csv", relation_affiliations)
append_csv("fields_of_study.csv", fields_of_study)
print(f"  {len(institutions)} institutions | {len(relation_affiliations)} affiliations")

# ═══════════════════════════════════════════════════════════════════════════════
# PDF CHUNK EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nEmbedding PDF chunks...")
pdf_datapoints = []
pdf_id_num = id_num
total_pdf_vectors = 0

pdf_files = [f for f in sorted(os.listdir(PDF_DIR)) if f.endswith(".pdf")]
print(f"  {len(pdf_files)} PDF files to process")

_t0 = time.time()
for idx, pdf_file in enumerate(pdf_files):
    paperId = pdf_file[:-4]
    try:
        full_text = read_pdf_text(PDF_DIR + pdf_file)
        text_chunks = chunk_text_by_tokens(full_text)
        embeddings = create_embedding(text_chunks)
        for chunk_idx, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
            pdf_datapoints.append({
                "id":           pdf_id_num,
                "paperId":      paperId,
                "chunk_text":   chunk,
                "chunk_index":  chunk_idx,
                "chunk_source": "pdf",
                "vector":       emb,
            })
            pdf_id_num += 1
    except Exception as e:
        pass  # skip corrupt PDFs silently

    if len(pdf_datapoints) >= PDF_BATCH_SIZE:
        res = client.insert(collection_name=COLLECTION, data=pdf_datapoints)
        total_pdf_vectors += res["insert_count"]
        pdf_datapoints.clear()

    progress_bar(idx + 1, len(pdf_files), "PDFs")

if pdf_datapoints:
    res = client.insert(collection_name=COLLECTION, data=pdf_datapoints)
    total_pdf_vectors += res["insert_count"]
    pdf_datapoints.clear()

print(f"\n  {total_pdf_vectors} PDF chunk vectors inserted. Total id_num: {pdf_id_num}")

# Update state with final id_num
with open(STATE_FILE, "w") as f:
    json.dump({"seen_paper_ids": list(seen_paper_ids),
               "paper_queue": paper_queue, "id_num": pdf_id_num}, f)

# ═══════════════════════════════════════════════════════════════════════════════
# NEO4J IMPORT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Importing into Neo4j...")
print(f"{'='*70}\n")

from neo4j import GraphDatabase

NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
BATCH = 500
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB — needed for full_text fields

def read_csv_rows(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def neo4j_batch(session, query, data, batch_size=BATCH):
    for i in range(0, len(data), batch_size):
        session.run(query, batch=data[i:i+batch_size])

# Clear graph
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
print("  Graph cleared.")

# Constraints
with driver.session() as session:
    for label in ["Paper", "Author", "Venue", "FieldOfStudy", "Institution"]:
        session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
print("  Constraints created.")

# Load CSVs
csv_papers     = read_csv_rows("papers.csv")
csv_authors    = read_csv_rows("authors.csv")
csv_venues     = read_csv_rows("venues.csv")
csv_fos        = read_csv_rows("fields_of_study.csv")
csv_insts      = read_csv_rows("institutions.csv")
csv_cites      = read_csv_rows("citations.csv")
csv_wrote      = read_csv_rows("written_by.csv")
csv_wt         = read_csv_rows("write_together.csv")
csv_pub_in     = read_csv_rows("written_for.csv")
csv_has_field  = read_csv_rows("field_of_study.csv")
csv_affil      = read_csv_rows("affiliations.csv")

print(f"  Loaded CSVs: {len(csv_papers)} papers, {len(csv_authors)} authors, "
      f"{len(csv_venues)} venues, {len(csv_fos)} fields, {len(csv_insts)} institutions")

# Import nodes
with driver.session() as session:
    neo4j_batch(session, """
        UNWIND $batch AS p
        MERGE (n:Paper {id: p.id})
        SET n.title          = p.title,
            n.abstract       = p.abstract,
            n.year           = toInteger(p.year),
            n.citationCount  = toInteger(p.citationCount),
            n.referenceCount = toInteger(p.referenceCount),
            n.isOpenAccess   = p.isOpenAccess,
            n.venue          = p.venue,
            n.externalIds    = p.externalIds,
            n.full_text      = p.full_text
    """, csv_papers)
    print(f"  Paper nodes: {len(csv_papers)}")

    neo4j_batch(session, """
        UNWIND $batch AS a
        MERGE (n:Author {id: a.authorId})
        SET n.name = a.name
    """, csv_authors)
    print(f"  Author nodes: {len(csv_authors)}")

    neo4j_batch(session, """
        UNWIND $batch AS v MERGE (n:Venue {id: v.venue})
    """, csv_venues)
    print(f"  Venue nodes: {len(csv_venues)}")

    neo4j_batch(session, """
        UNWIND $batch AS f MERGE (n:FieldOfStudy {id: f.fieldOfStudy})
    """, csv_fos)
    print(f"  FieldOfStudy nodes: {len(csv_fos)}")

    neo4j_batch(session, """
        UNWIND $batch AS i MERGE (n:Institution {id: i.institution})
    """, csv_insts)
    print(f"  Institution nodes: {len(csv_insts)}")

# Import relationships
with driver.session() as session:
    neo4j_batch(session, """
        UNWIND $batch AS r
        MATCH (a:Paper {id: r.cites})
        MATCH (b:Paper {id: r.cited_by})
        MERGE (a)-[:CITES]->(b)
    """, csv_cites)
    print(f"  CITES: {len(csv_cites)}")

    neo4j_batch(session, """
        UNWIND $batch AS r
        MERGE (a:Author {id: r.written_by})
        MERGE (p:Paper {id: r.writes})
        MERGE (a)-[:WROTE]->(p)
    """, csv_wrote)
    print(f"  WROTE: {len(csv_wrote)}")

    neo4j_batch(session, """
        UNWIND $batch AS r
        MERGE (a:Author {id: r.author1})
        MERGE (b:Author {id: r.author2})
        MERGE (a)-[:WRITES_WITH]->(b)
    """, csv_wt)
    print(f"  WRITES_WITH: {len(csv_wt)}")

    neo4j_batch(session, """
        UNWIND $batch AS r
        MERGE (p:Paper {id: r.paperId})
        MERGE (v:Venue {id: r.venue})
        MERGE (p)-[:PUBLISHED_IN]->(v)
    """, csv_pub_in)
    print(f"  PUBLISHED_IN: {len(csv_pub_in)}")

    neo4j_batch(session, """
        UNWIND $batch AS r
        MERGE (p:Paper {id: r.paperId})
        MERGE (f:FieldOfStudy {id: r.fieldOfStudy})
        MERGE (p)-[:HAS_FIELD]->(f)
    """, csv_has_field)
    print(f"  HAS_FIELD: {len(csv_has_field)}")

    neo4j_batch(session, """
        UNWIND $batch AS r
        MERGE (a:Author {id: r.authorId})
        MERGE (i:Institution {id: r.affiliation})
        MERGE (a)-[:AFFILIATED_WITH]->(i)
    """, csv_affil)
    print(f"  AFFILIATED_WITH: {len(csv_affil)}")

driver.close()
print(f"\n{'='*70}")
print("ALL DONE — ingestion + Neo4j import complete!")
print(f"{'='*70}")
