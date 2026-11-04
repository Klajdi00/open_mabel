"""fetch_egodex.py — pull only the pose HDF5 files out of Apple's EgoDex zips (no video).

EgoDex (Hoque et al. 2025, arXiv:2505.11709) is 829 h of Apple Vision Pro recordings: head (camera),
wrists, fingers and arm joints at 30 Hz. The zips are mostly MP4; this reads the zip's central
directory over HTTP range requests and extracts the *.hdf5 members only (~1-2 % of the bytes).

    python fetch_egodex.py test            # 17 GB zip -> pose files only
    python fetch_egodex.py part1 --max 3000
Output: data/raw/egodex/<split>/<task>/<n>.hdf5
"""
import argparse, io, os, struct, sys, urllib.request, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor

URL = "https://ml-site.cdn-apple.com/datasets/egodex/{}.zip"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "egodex")


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 4: raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", help="test | extra | part1..part5")
    ap.add_argument("--max", type=int, default=0, help="stop after this many hdf5 files (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    url = URL.format(a.split)
    hf = HttpFile(url); size = hf.size
    names = [i for i in zipfile.ZipFile(io.BufferedReader(hf, 1 << 20)).infolist() if i.filename.endswith(".hdf5")]
    if a.max: names = names[:a.max]
    print(f"{a.split}: {len(names)} hdf5 files, {sum(i.compress_size for i in names) / 1e9:.2f} GB", flush=True)

    def get(info):
        rel = info.filename.split("/", 1)[1] if "/" in info.filename else info.filename
        dst = os.path.join(OUT, a.split, rel)
        if os.path.exists(dst) and os.path.getsize(dst) == info.file_size: return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        f = HttpFile.__new__(HttpFile); f.url, f.size, f.pos = url, size, info.header_offset
        head = f.read(30)                                   # local file header: name/extra lengths may differ from the CD
        n, m = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + n + m)
        raw = f.read(info.compress_size)
        data = raw if info.compress_type == 0 else zlib.decompress(raw, -15)
        with open(dst + ".part", "wb") as g: g.write(data)
        os.replace(dst + ".part", dst)
        return 1

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(get, names)):
            done += 1
            if done % 200 == 0: print(f"  {done}/{len(names)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
