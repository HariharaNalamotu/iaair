"""
Expand ground truth dataset from 100 to 250 papers.
- Reads existing ground_truth_papers.csv (100 papers)
- Reads papers.csv, excludes the 100 already sampled
- Randomly samples 150 more papers (seed=42)
- Outputs new_papers_150.csv and ground_truth_papers_250.csv
"""

import pandas as pd

# 1. Load existing ground truth
gt = pd.read_csv("/Users/Hari/IAAIR/ground_truth_papers.csv")
existing_ids = set(gt["paperId"].tolist())
print(f"Existing ground truth papers: {len(gt)}")

# 2. Load full papers dataset
papers = pd.read_csv("/Users/Hari/IAAIR/papers.csv")
print(f"Total papers in papers.csv: {len(papers)}")

# 3. Exclude already-sampled papers
candidates = papers[~papers["id"].isin(existing_ids)].copy()
print(f"Candidate papers (after exclusion): {len(candidates)}")

# 4. Sample 150 new papers with seed 42
new_sample = candidates.sample(n=150, random_state=42)

# 5. Format output columns: paperId, title, abstract (truncated 500 chars), year, venue
new_out = pd.DataFrame({
    "paperId": new_sample["id"].values,
    "title": new_sample["title"].values,
    "abstract": new_sample["abstract"].fillna("").str[:500].values,
    "year": new_sample["year"].values,
    "venue": new_sample["venue"].values,
})

new_out.to_csv("/Users/Hari/IAAIR/new_papers_150.csv", index=False)
print(f"Wrote new_papers_150.csv with {len(new_out)} papers")

# 6. Truncate existing ground truth abstracts to 500 chars for consistency
gt["abstract"] = gt["abstract"].fillna("").str[:500]

# 7. Combine into 250-paper dataset
combined = pd.concat([gt, new_out], ignore_index=True)
combined.to_csv("/Users/Hari/IAAIR/ground_truth_papers_250.csv", index=False)
print(f"Wrote ground_truth_papers_250.csv with {len(combined)} papers")

# Sanity checks
assert len(combined) == 250, f"Expected 250, got {len(combined)}"
assert combined["paperId"].nunique() == 250, "Duplicate paper IDs found!"
print("All checks passed.")
