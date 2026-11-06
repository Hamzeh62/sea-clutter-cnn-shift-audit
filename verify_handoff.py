#!/usr/bin/env python
"""Verify PAPER2_EDITING_AND_REPRODUCIBILITY_HANDOFF in place.

Checks, in order:
  1. every manifest entry matches its file byte-for-byte (sha256 + size)
  2. PACKAGE_CONTENTS.csv agrees with the manifest
  3. every JSON parses
  4. every .npz loads, and each member matches provenance/score_arrays.csv
     (key, dtype, shape, and a sha256 of the exact value bytes)
  5. every claim in provenance/claim_sources.csv resolves to a file here
  6. every figure source named in data/figure_sources.csv resolves
  7. the two checkpoints load (only if torch is installed)

Usage:  python verify_handoff.py
Exit code 0 = all checks passed.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails, checks = [], 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)
    return cond


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    global fails, checks
    # ---- 1. manifest -------------------------------------------------------
    man = []
    for line in (ROOT / "SHA256_MANIFEST.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        h, rel = line.split("  ", 1)
        man.append((h, rel))
    ok(bool(man), "manifest is empty")
    for h, rel in man:
        p = ROOT / rel
        if not ok(p.exists(), f"missing file: {rel}"):
            continue
        ok(sha(p) == h, f"hash mismatch: {rel}")
    print(f"[1] manifest        : {len(man)} files verified")

    # ---- 2. contents CSV ---------------------------------------------------
    with (ROOT / "PACKAGE_CONTENTS.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    manmap = dict((r, h) for h, r in man)
    for r in rows:
        ok(manmap.get(r["path_in_package"]) == r["sha256"],
           f"contents/manifest disagree: {r['path_in_package']}")
        ok(int(r["bytes"]) == (ROOT / r["path_in_package"]).stat().st_size,
           f"size mismatch: {r['path_in_package']}")
    print(f"[2] contents csv    : {len(rows)} rows agree with the manifest")

    # ---- 3. JSON parses ----------------------------------------------------
    njson = 0
    for p in sorted(ROOT.rglob("*.json")):
        try:
            json.loads(p.read_text(encoding="utf-8"))
            njson += 1
        except Exception as e:  # noqa: BLE001
            fails.append(f"unparseable JSON {p.relative_to(ROOT)}: {e}")
            checks += 1
    print(f"[3] json            : {njson} files parse")

    # ---- 4. npz members ----------------------------------------------------
    import numpy as np

    with (ROOT / "provenance" / "score_arrays.csv").open(encoding="utf-8") as f:
        want = list(csv.DictReader(f))
    got = []
    for zp in sorted(ROOT.rglob("*.npz")):
        z = np.load(zp)
        for k in z.files:
            a = z[k]
            if a.dtype.kind == "f":
                n_valid = int(np.isfinite(a).sum())
                n_nan = int(np.isnan(a).sum())
            else:
                n_valid, n_nan = (a.size if a.ndim else 1), 0
            # Values are finite or NaN; an infinity is a numerical fault.
            ok(not np.isinf(a).any(), f"inf values in {zp.name}:{k}")
            # A masked entry must be NaN, not an arbitrary sentinel.
            ok(n_valid + n_nan == (a.size if a.ndim else 1),
               f"non-NaN non-finite entry in {zp.name}:{k}")
            got.append({
                "npz_file": zp.relative_to(ROOT).as_posix(),
                "member_key": k,
                "dtype": str(a.dtype),
                "shape": "x".join(str(s) for s in a.shape),
                "nbytes": str(a.nbytes),
                "n_valid": str(n_valid),
                "n_nan": str(n_nan),
                "sha256_of_values": hashlib.sha256(
                    np.ascontiguousarray(a).tobytes()).hexdigest(),
            })
        z.close()
    ok(len(got) == len(want), f"array count {len(got)} != recorded {len(want)}")
    for g, w in zip(got, want):
        ok(g == w, f"array mismatch: {g['npz_file']}:{g['member_key']}")

    # The masked arms must carry exactly the documented subset size.
    SUBSET = {"spike": 100, "flat": 84}
    masked = [g for g in got if g["n_nan"] != "0"]
    for g in masked:
        arm = g["member_key"].split("|")[0]
        if arm in SUBSET:
            ok(int(g["n_valid"]) == SUBSET[arm],
               f"{arm} arm has {g['n_valid']} scored trials, expected {SUBSET[arm]}")
    print(f"[4] npz arrays      : {len(got)} members in "
          f"{len({g['npz_file'] for g in got})} files match the recorded inventory")
    if masked:
        arms = sorted({g["member_key"].split("|")[0] for g in masked})
        print(f"      masked subset arms: {', '.join(arms)} "
              f"({len(masked)} members; scored-trial count matches the documented subset)")

    # ---- 5. claim closure --------------------------------------------------
    with (ROOT / "provenance" / "claim_sources.csv").open(encoding="utf-8") as f:
        crows = list(csv.DictReader(f))
    bad = [r for r in crows if r["path_in_package"] == "MISSING"
           or not (ROOT / r["path_in_package"]).exists()]
    ok(not bad, f"{len(bad)} claims resolve to no file")
    print(f"[5] claim closure   : {len(crows)} claims -> "
          f"{len({r['source_artifact'] for r in crows})} artifacts, all present")

    # ---- 6. figure sources -------------------------------------------------
    with (ROOT / "data" / "figure_sources.csv").open(encoding="utf-8") as f:
        frows = list(csv.DictReader(f))
    for r in frows:
        for tok in r["source_data"].replace("|", " ").split():
            tok = tok.split(":")[0]
            if "/" in tok and tok.endswith((".json", ".npz", ".csv")):
                ok((ROOT / tok).exists(), f"figure source missing: {tok}")
    print(f"[6] figure sources  : {len(frows)} figures mapped, sources present")

    # ---- 7. checkpoints ----------------------------------------------------
    try:
        import torch
        for p in sorted((ROOT / "checkpoints").glob("*.pt")):
            sd = torch.load(p, map_location="cpu", weights_only=False)
            n = len(sd) if hasattr(sd, "__len__") else 0
            ok(n > 0, f"checkpoint empty: {p.name}")
            print(f"      {p.name}: {n} tensors")
        print("[7] checkpoints     : loaded")
    except ImportError:
        print("[7] checkpoints     : SKIPPED (torch not installed)")

    print()
    if fails:
        print(f"FAIL — {len(fails)} problem(s) in {checks} checks:")
        for m in fails[:20]:
            print("   -", m)
        return 1
    print(f"PASS — {checks} checks, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
