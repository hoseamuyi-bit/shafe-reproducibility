"""Select source images reproducibly from the official DIV2K training archive."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import time
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile
import zlib


HERE = Path(__file__).resolve().parent
URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
EXPECTED_BYTES = 3530603713
SEED = 20261123
COUNT = 108


def request_range(first, size):
    expected = f"bytes {first}-{first + size - 1}/{EXPECTED_BYTES}"
    headers = {"Range": f"bytes={first}-{first + size - 1}", "Accept-Encoding": "identity"}
    for attempt in range(4):
        try:
            with urlopen(Request(URL, headers=headers), timeout=35) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise OSError(f"Unexpected range response for {expected}")
                data = response.read(size + 1)
            if len(data) != size:
                raise OSError(f"Truncated range {expected}")
            return data
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class ArchiveIndex(io.RawIOBase):
    def __init__(self):
        self.offset = 0

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.offset, io.SEEK_END: EXPECTED_BYTES}
        self.offset = base[whence] + offset
        return self.offset

    def read(self, size=-1):
        available = EXPECTED_BYTES - self.offset
        count = min(size, available) if size >= 0 else available
        if count <= 0:
            return b""
        if count > 1024 * 1024:
            raise ValueError("ZIP index read exceeded its expected size")
        data = request_range(self.offset, count)
        self.offset += count
        return data


def download(info):
    path = HERE / "holdout_source" / Path(info.filename).name
    data = path.read_bytes() if path.exists() else b""
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        if info.compress_type not in (0, ZIP_DEFLATED):
            raise ValueError(f"Unsupported compression in {info.filename}")
        header = request_range(info.header_offset, 30)
        sizes = struct.unpack("<4s5H3I2H", header)
        if sizes[0] != b"PK\x03\x04":
            raise ValueError(f"Invalid local header for {info.filename}")
        payload_start = info.header_offset + len(header) + sizes[-2] + sizes[-1]
        payload = request_range(payload_start, info.compress_size)
        data = zlib.decompress(payload, -15) if info.compress_type == ZIP_DEFLATED else payload
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError(f"Bad ZIP CRC/size for {info.filename}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Expected PNG for {info.filename}")
    width, height = struct.unpack(">II", data[16:24])
    return dict(member=info.filename, path=str(path.resolve()), width=width, height=height,
                sha256=hashlib.sha256(data).hexdigest(), crc32=f"{info.CRC:08x}")


def main():
    with ZipFile(ArchiveIndex()) as archive:
        members = sorted((x for x in archive.infolist() if x.filename.lower().endswith(".png")),
                         key=lambda x: x.filename)
    if len(members) != 800:
        raise ValueError(f"Expected 800 source images, got {len(members)}")
    random.Random(SEED).shuffle(members)
    (HERE / "holdout_source").mkdir(exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(download, members[:COUNT]), 1):
            results.append(result)
            if i % 12 == 0:
                print(f"Verified official DIV2K training archive images: {i}/{COUNT}", flush=True)
    path = HERE / "holdout_source.json"
    path.write_text(json.dumps(dict(url=URL, archive_bytes=EXPECTED_BYTES, seed=SEED, records=results),
                               indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
