"""Download the matching 0731 DSpark sidecar, pinned and SHA-256 verified."""
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import requests

REVISION = "fbbb5b93fb787c21338159b0af3318bb3f4d9768"
FILENAME = "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
SHA256 = "2c7ac54b0b64a99df1f139a9f1371a00198265e1d6a614b77597d20a655a4249"
SIZE = 10896057440
URL = f"https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/{REVISION}/{FILENAME}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16, choices=range(1, 65))
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "models/deepseek4-dspark" / FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".gguf.part")
    source = destination if destination.exists() else partial
    if not destination.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > SIZE:
            raise RuntimeError(f"Oversized partial download: {partial}")
        # Bound RAM and preserve completed ranges across interruption. The
        # original contiguous prefix is retained, too. Only join after all
        # ranges arrive; the final digest checks the complete assembled file.
        chunk_size = 16 * 1024 * 1024
        ranges = [(start, min(start + chunk_size, SIZE)) for start in range(offset, SIZE, chunk_size)]
        def fetch(bounds):
            start, end = bounds
            path = destination.with_suffix(f".range-{start}-{end}")
            if path.exists() and path.stat().st_size == end - start:
                return path
            for attempt in range(4):
                try:
                    with requests.get(URL, headers={"Range": f"bytes={start}-{end - 1}"}, stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{SIZE}":
                            raise RuntimeError("Server did not honor download range")
                        with path.open("wb") as output:
                            for chunk in response.iter_content(1024 * 1024):
                                output.write(chunk)
                    if path.stat().st_size != end - start:
                        raise RuntimeError("Incomplete range")
                    return path
                except (requests.RequestException, RuntimeError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        completed, last = offset, time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
                completed += future.result().stat().st_size
                if time.monotonic() - last > 15:
                    print(f"Downloaded {completed / SIZE:.1%} ({completed / 1024**3:.2f} GiB)", flush=True)
                    last = time.monotonic()
        with partial.open("ab") as output:
            for start, end in ranges:
                path = destination.with_suffix(f".range-{start}-{end}")
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        output.write(chunk)
    if source.stat().st_size != SIZE:
        raise RuntimeError("Incomplete DSpark download")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError(f"SHA-256 mismatch: {source}; file not activated")
    if source == partial:
        partial.replace(destination)
    # These are this downloader's fixed-name scratch ranges, within models/.
    for path in destination.parent.glob(destination.stem + ".range-*-*"):
        if path.resolve().parent == destination.parent.resolve():
            path.unlink()
    print(f"Verified {SHA256}\n{destination}", flush=True)


if __name__ == "__main__":
    main()
