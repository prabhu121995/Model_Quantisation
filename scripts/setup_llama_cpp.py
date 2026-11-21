"""Download llama.cpp (prebuilt Windows CPU binaries + matching source for the converter).

Run once:  uv run scripts/setup_llama_cpp.py            (default release below)
           uv run scripts/setup_llama_cpp.py b12700     (a newer release tag)
"""

import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile

from model_quantization import config

TAG = sys.argv[1] if len(sys.argv) > 1 else "b12602"  # tested with this release
BASE = "https://github.com/ggml-org/llama.cpp"


def fetch(url: str) -> bytes:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as r:
        return r.read()


def main() -> None:
    bin_dir, src_dir = config.LLAMA_CPP_BIN, config.LLAMA_CPP_SRC
    for d in (bin_dir, src_dir):
        if d.exists():
            shutil.rmtree(d)

    zipfile.ZipFile(io.BytesIO(fetch(f"{BASE}/releases/download/{TAG}/llama-{TAG}-bin-win-cpu-x64.zip"))).extractall(bin_dir)

    with tarfile.open(fileobj=io.BytesIO(fetch(f"{BASE}/archive/refs/tags/{TAG}.tar.gz"))) as tar:
        tar.extractall(src_dir.parent, filter="data")
    (src_dir.parent / f"llama.cpp-{TAG}").rename(src_dir)

    print(f"Done.\n  binaries: {bin_dir}\n  source  : {src_dir}")


if __name__ == "__main__":
    main()
