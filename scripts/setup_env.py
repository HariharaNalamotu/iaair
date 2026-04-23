#!/usr/bin/env python3
"""
IAAIR one-time environment setup for Windows.
Run from a Windows terminal (PowerShell or CMD):

    C:\\Users\\harih\\anaconda3\\python.exe scripts\\setup_env.py

Steps:
  1. Install Java 21 via winget
  2. Download and install Neo4j 5 Community for Windows
  3. Set password and restore database from transfer/neo4j.dump
  4. Install Neo4j as a Windows service and start it
  5. Create project venv at .venv/
  6. Install PyTorch (CUDA 12.8), faiss-cpu, and all project deps
  7. Pre-compute paper embeddings on GPU
  8. Verify everything works

Note: Run scripts/extract_vectors.py once in WSL2 to populate data/vectors.npy
      before running the evaluation. That is the only WSL2 step.
"""
import os, pathlib, subprocess, sys, time, urllib.request, zipfile, shutil

ROOT       = pathlib.Path(__file__).parent.parent
VENV       = ROOT / ".venv"
PY         = str(VENV / "Scripts" / "python.exe")
PIP        = str(VENV / "Scripts" / "pip.exe")
DUMP_DIR   = ROOT / "transfer"
NEO4J_DIR  = pathlib.Path("C:/neo4j")
NEO4J_PASS = "Thammu123"
BASE_PY    = str(pathlib.Path(sys.executable).parent / "python.exe")

def banner(msg): print(f"\n{'='*60}\n==> {msg}\n{'='*60}", flush=True)
def ok(msg):     print(f"  [OK]   {msg}", flush=True)
def warn(msg):   print(f"  [WARN] {msg}", flush=True)
def run(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check)
def out(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

if sys.platform != "win32":
    sys.exit("Run this script from Windows, not WSL2.\n"
             "For WSL2 (Neo4j only), see: scripts/setup_env.py is Windows-only.")

# ── 1. Java 21 ────────────────────────────────────────────────────────────────
banner("Step 1/8 — Java 21")
v = out("java -version 2>&1")
if any(f'"{n}.' in v for n in range(21, 30)):
    ok(f"Already installed: {v.splitlines()[0]}")
else:
    print("  Installing Java 21 via winget...", flush=True)
    r = run("winget install --id Microsoft.OpenJDK.21 "
            "--accept-source-agreements --accept-package-agreements --silent",
            check=False)
    # Reload PATH
    os.environ["PATH"] = (
        os.environ.get("PATH", "") + ";"
        + r"C:\Program Files\Microsoft\jdk-21.0.0+35\bin"
    )
    ok(out("java -version 2>&1").splitlines()[0])

# ── 2. Neo4j 5 Community for Windows ─────────────────────────────────────────
banner("Step 2/8 — Neo4j 5 Community")
if (NEO4J_DIR / "bin" / "neo4j-admin.bat").exists():
    ok(f"Already installed at {NEO4J_DIR}")
else:
    versions = ["5.26.0", "5.25.1", "5.24.2", "5.23.0"]
    installed = False
    for v in versions:
        url     = f"https://dist.neo4j.org/neo4j-community-{v}-windows.zip"
        zippath = pathlib.Path(os.environ["TEMP"]) / f"neo4j-{v}.zip"
        print(f"  Downloading Neo4j {v} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, zippath)
            print("  Extracting ...", flush=True)
            with zipfile.ZipFile(zippath, "r") as z:
                z.extractall("C:/")
            extracted = next(
                (p for p in pathlib.Path("C:/").iterdir()
                 if p.name.startswith(f"neo4j-community-{v}")), None
            )
            if extracted:
                if NEO4J_DIR.exists():
                    shutil.rmtree(NEO4J_DIR)
                extracted.rename(NEO4J_DIR)
                zippath.unlink()
                installed = True
                ok(f"Neo4j {v} installed at {NEO4J_DIR}")
                break
        except Exception as e:
            warn(f"  Failed for {v}: {e}")
    if not installed:
        sys.exit("Could not download Neo4j. Check internet connection.")

ADMIN = str(NEO4J_DIR / "bin" / "neo4j-admin.bat")
NEO4J = str(NEO4J_DIR / "bin" / "neo4j.bat")

# ── 3. Restore database ───────────────────────────────────────────────────────
banner("Step 3/8 — Restore Neo4j database")
# Stop service if running
run(f'net stop Neo4j 2>nul', check=False)
time.sleep(2)

run(f'"{ADMIN}" dbms set-initial-password {NEO4J_PASS}', check=False)

dump = DUMP_DIR / "neo4j.dump"
if not dump.exists():
    sys.exit(f"Dump not found: {dump}")

r = run(f'"{ADMIN}" database load neo4j '
        f'--from-path="{DUMP_DIR}" --overwrite-destination=true', check=False)
if r.returncode != 0:
    warn("5.x syntax failed, trying 4.x...")
    run(f'"{ADMIN}" load --database=neo4j --from="{dump}" --force')
ok("Database restored.")

# ── 4. Install and start Neo4j Windows service ────────────────────────────────
banner("Step 4/8 — Neo4j Windows service")
run(f'"{NEO4J}" install-service', check=False)
run('net start Neo4j', check=False)
print("  Waiting for Neo4j (up to 60 s)...", flush=True)
for i in range(30):
    time.sleep(2)
    svc = out("sc query Neo4j")
    if "RUNNING" in svc:
        ok("Neo4j service is running.")
        break
ok("Neo4j started.")

# ── 5. Create project venv ───────────────────────────────────────────────────
banner(f"Step 5/8 — Project venv at {VENV}")
if (VENV / "Scripts" / "python.exe").exists():
    ok(f"venv already exists ({out(PY + ' --version')})")
else:
    run(f'"{BASE_PY}" -m venv "{VENV}"')
    ok(f"Created: {out(PY + ' --version')}")

run(f'"{PIP}" install --quiet --upgrade pip')

# ── 6. Install packages ───────────────────────────────────────────────────────
banner("Step 6/8 — Install packages")
print("  PyTorch (CUDA 12.8) — this may take a few minutes...", flush=True)
run(f'"{PIP}" install --quiet torch torchvision torchaudio '
    '--index-url https://download.pytorch.org/whl/cu128')
ok(out(f'"{PY}" -c "import torch; '
       'print(torch.__version__, torch.cuda.is_available())"'))

run(f'"{PIP}" install --quiet faiss-cpu')
ok(out(f'"{PY}" -c "import faiss; print(\'faiss\', faiss.__version__)"'))

run(f'"{PIP}" install --quiet '
    '"sentence-transformers>=5.0" "neo4j>=5.0" '
    'python-dotenv numpy matplotlib tiktoken pypdf '
    'requests transformers huggingface-hub')
ok("All packages installed.")

# ── 7. Pre-compute paper embeddings ──────────────────────────────────────────
banner("Step 7/8 — Pre-compute paper embeddings (GPU)")
if (ROOT / "data" / "paper_embeddings.npy").exists():
    ok("paper_embeddings.npy already exists — skipping.")
else:
    run(f'"{PY}" "{ROOT / "scripts" / "precompute_embeddings.py"}"')

# ── 8. Verify ─────────────────────────────────────────────────────────────────
banner("Step 8/8 — Verify")
run(f'"{PY}" "{ROOT / "scripts" / "verify_setup.py"}"')

print(f"""
{'='*60}
 Setup complete!

 NEXT: extract FAISS vectors (run ONCE in WSL2):
   ~/iaair-venv/bin/python /mnt/c/Users/harih/hybrid-graphrag/scripts/extract_vectors.py
   (or use any WSL2 Python with pymilvus[milvus_lite] installed)

 Then run evaluation from Windows:
   .venv\\Scripts\\python.exe evaluation\\run_evaluation.py
{'='*60}
""")
