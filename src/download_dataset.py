"""Download and verify the AutoScout24 dataset from Zenodo.

Re-runnable: a valid existing file (size + MD5) is never overwritten or re-downloaded.
Usage:  python -m src.download_dataset
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date

import requests

from src import config

TIMEOUT = (15, 60)  # (connect, read) seconds
MAX_ATTEMPTS = 3
CHUNK = 1024 * 1024


def md5_of_file(path, chunk_size: int = CHUNK) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def fetch_remote_metadata() -> dict:
    """Read file name / size / md5 from the Zenodo API (not from our hard-coded copy)."""
    r = requests.get(config.ZENODO_API_URL, timeout=TIMEOUT)
    r.raise_for_status()
    rec = r.json()
    files = {f["key"]: f for f in rec["files"]}
    if config.RAW_FILENAME not in files:
        raise RuntimeError(f"File {config.RAW_FILENAME} not found in Zenodo record; found {list(files)}")
    f = files[config.RAW_FILENAME]
    return {
        "name": f["key"],
        "size": f["size"],
        "md5": f["checksum"].split(":", 1)[1],
        "url": f["links"]["self"],
        "version": rec["metadata"].get("version"),
        "license": (rec["metadata"].get("license") or {}).get("id"),
        "publication_date": rec["metadata"].get("publication_date"),
        "doi": rec.get("doi"),
    }


def is_valid(path, size: int, md5: str) -> bool:
    return path.exists() and path.stat().st_size == size and md5_of_file(path) == md5


def download(url: str, dest, expected_size: int) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as r:
                if r.status_code == 200 and done:  # server ignored Range
                    done = 0
                r.raise_for_status()
                mode = "ab" if done and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if (done // CHUNK) % 50 == 0:
                            print(f"  {done / 1e6:8.1f} / {expected_size / 1e6:.1f} MB", flush=True)
            part.replace(dest)  # rename only after the download is complete
            return
        except (requests.RequestException, OSError) as e:
            last_err = e
            print(f"\n  Attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
    raise RuntimeError(f"Download failed after {MAX_ATTEMPTS} attempts: {last_err}")


def manual_instructions() -> str:
    return (
        f"Tải thủ công từ {config.ZENODO_RECORD_URL} (file {config.RAW_FILENAME}) "
        f"và đặt vào {config.DATA_RAW}. MD5 mong đợi: {config.PUBLISHED_MD5}."
    )


def main() -> int:
    dest = config.RAW_PATH
    try:
        meta = fetch_remote_metadata()
    except Exception as e:  # network down: fall back to published constants
        print(f"[WARN] Cannot read Zenodo metadata ({e}); using published constants.")
        meta = {"size": config.PUBLISHED_SIZE_BYTES, "md5": config.PUBLISHED_MD5,
                "url": config.ZENODO_FILE_URL, "version": config.DATASET_VERSION,
                "license": "mit-license", "publication_date": "2025-11-18", "doi": config.ZENODO_DOI}

    if dest.exists():
        print(f"File exists: {dest} -> verifying size and MD5 ...")
        if is_valid(dest, meta["size"], meta["md5"]):
            print("[OK] Existing file is valid; no download needed.")
        else:
            print("[ERROR] Existing file is NOT valid (size/MD5 mismatch). It was left untouched.")
            print("        Xoá hoặc đổi tên file đó rồi chạy lại để tải mới.")
            return 1
    else:
        print(f"Downloading {meta['url']}")
        try:
            download(meta["url"], dest, meta["size"])
        except Exception as e:
            print(f"[ERROR] {e}\n{manual_instructions()}")
            return 1
        if not is_valid(dest, meta["size"], meta["md5"]):
            bad = dest.with_suffix(dest.suffix + ".corrupt")
            dest.replace(bad)
            print(f"[ERROR] Checksum/size mismatch; moved to {bad}.\n{manual_instructions()}")
            return 1
        print("[OK] Download complete and MD5 verified.")

    info = {
        "source_url": config.ZENODO_RECORD_URL,
        "doi": meta["doi"],
        "file_name": config.RAW_FILENAME,
        "size_bytes": dest.stat().st_size,
        "md5_verified": md5_of_file(dest),
        "md5_published": meta["md5"],
        "access_date": date.today().isoformat(),
        "dataset_version": meta["version"],
        "publication_date": meta["publication_date"],
        "license": meta["license"],
    }
    out = config.DATA_RAW / "SOURCE.json"
    if out.exists():  # keep the original access date when the same (verified) file is re-checked
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("md5_verified") == info["md5_verified"]:
            info["access_date"] = old.get("access_date", info["access_date"])
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
