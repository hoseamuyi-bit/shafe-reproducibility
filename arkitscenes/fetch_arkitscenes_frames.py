"""Extract selected ARKitScenes frames from a published ZIP over HTTP Range."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import struct
from time import sleep
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib


BASE = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1'
LOCAL_HEADER = struct.Struct('<4s5H3I2H')
TAIL_BYTES = 4 * 1024 * 1024


def request_bytes(url, selection, total=None, etag=None):
    for attempt in range(3):
        try:
            request = Request(url, headers={'Range': f'bytes={selection}', 'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=30) as response:
                if response.status != 206:
                    raise ValueError(f'Expected HTTP 206 for {selection}, received {response.status}')
                content_range = response.headers['Content-Range']
                start, stop, length = map(int, re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range).groups())
                if total is not None and (length != total or selection != f'{start}-{stop}'):
                    raise ValueError(f'Unexpected Content-Range: {content_range} for {selection}')
                if etag and response.headers['ETag'] != etag:
                    raise ValueError(f'ETag changed for {url}')
                payload = response.read(stop - start + 2)
                if len(payload) != stop - start + 1:
                    raise ValueError(f'Truncated range: {content_range}')
                return payload, content_range, response.headers['ETag']
        except (OSError, ValueError):
            if attempt == 2:
                raise
            sleep(2 ** attempt)


def atomic_write(path, data):
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_bytes(data)
    partial.replace(path)


class FrameArchive:
    def __init__(self, root, split, asset):
        video_id = root.name.removeprefix('arkit_')
        self.url = f'{BASE}/raw/{split}/{video_id}/{asset}.zip'
        self.root = root / asset
        self.root.mkdir(exist_ok=True)
        tail = root / f'{asset}_tail.bin'
        header = tail.with_suffix('.headers')
        if asset == 'vga_wide' and not tail.exists() and (root / 'vga_tail.bin').exists():
            tail = root / 'vga_tail.bin'
            header = root / 'vga_tail.headers'
        if asset == 'lowres_wide' and not tail.exists() and (root / 'lowres_tail.bin').exists():
            tail = root / 'lowres_tail.bin'
            header = root / 'lowres_tail.bin.headers'
        if not (tail.exists() and header.exists()):
            payload, content_range, etag = request_bytes(self.url, f'-{TAIL_BYTES}')
            atomic_write(tail, payload)
            atomic_write(header, f'Content-Range: {content_range}\nETag: {etag}\n'.encode())
        headers = header.read_text(encoding='ascii')
        match = re.search(r'^content-range: bytes (\d+)-(\d+)/(\d+)', headers, re.MULTILINE | re.IGNORECASE)
        self.tail_start, stop, self.total = map(int, match.groups())
        self.etag = re.search(r'^etag:\s*(.+)', headers, re.MULTILINE | re.IGNORECASE)[1].strip()
        payload = tail.read_bytes()
        if len(payload) != stop - self.tail_start + 1 or stop + 1 != self.total:
            raise ValueError(f'Invalid ZIP tail for {self.url}')
        self.tail_sha256 = sha256(payload).hexdigest()
        with ZipFile(BytesIO(payload)) as archive:
            entries = sorted(archive.infolist(), key=lambda info: info.header_offset)
            endpoints = [info.header_offset for info in entries[1:]] + [archive.start_dir]
            self.frames = {
                float(Path(info.filename).stem.rsplit('_', 1)[1]): (info, self.tail_start + end)
                for info, end in zip(entries, endpoints, strict=True)
                if info.filename.endswith('.png')
            }

    def extract(self, timestamp):
        info, stop = self.frames[timestamp]
        name = Path(info.filename).name
        destination = self.root / name
        start = self.tail_start + info.header_offset
        if destination.exists():
            payload = destination.read_bytes()
        else:
            selection = f'{start}-{stop - 1}'
            member, _, _ = request_bytes(self.url, selection, self.total, self.etag)
            magic, _, flags, method, *_, name_len, extra_len = LOCAL_HEADER.unpack_from(member)
            if magic != b'PK\x03\x04' or flags & 1 or method != info.compress_type:
                raise ValueError(f'Invalid local ZIP header for {name}')
            encoded_name = member[LOCAL_HEADER.size:LOCAL_HEADER.size + name_len]
            if encoded_name.decode('utf-8' if flags & 2048 else 'cp437') != info.filename:
                raise ValueError(f'Wrong ZIP member name for {name}')
            data_start = LOCAL_HEADER.size + name_len + extra_len
            compressed = member[data_start:data_start + info.compress_size]
            if method == ZIP_STORED:
                payload = compressed
            elif method == ZIP_DEFLATED:
                payload = zlib.decompress(compressed, wbits=-15)
            else:
                raise ValueError(f'Unsupported ZIP compression for {name}: {method}')
        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
            raise ValueError(f'ZIP size or CRC mismatch for {name}')
        if not destination.exists():
            atomic_write(destination, payload)
        return dict(timestamp=timestamp, member=info.filename, bytes=len(payload),
                    range_start=start, range_stop_inclusive=stop - 1,
                    crc32=f'{info.CRC:08x}', sha256=sha256(payload).hexdigest())


def official_hr_times(root):
    video_id = root.name.removeprefix('arkit_')
    with ZipFile(root / f'{video_id}.zip') as archive:
        return sorted(float(Path(name).stem.rsplit('_', 1)[1]) for name in archive.namelist()
                      if '/highres_depth/' in name and name.endswith('.png'))


def download_frames(archive, times):
    times = sorted(set(times))
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(archive.extract, times))
    manifest = dict(url=archive.url, etag=archive.etag, total_zip_bytes=archive.total,
                    zip_tail_sha256=archive.tail_sha256, frames=records)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('asset', choices=('vga_wide', 'lowres_wide', 'lowres_depth', 'wide'))
    parser.add_argument('times', type=float, nargs='*')
    parser.add_argument('--split', default='Validation', choices=('Validation', 'Training'))
    args = parser.parse_args()
    archive = FrameArchive(args.root, args.split, args.asset)
    times = args.times
    if not times:
        times = []
        for time in official_hr_times(args.root):
            if args.asset == 'vga_wide':
                times.extend(sorted(archive.frames, key=lambda frame: abs(frame - time))[:2])
            elif time in archive.frames:
                times.append(time)
    manifest = download_frames(archive, times)
    path = args.root / f'{args.asset}_frames.json'
    atomic_write(path, (json.dumps(manifest, indent=2) + '\n').encode())
    print(f'{args.asset}: {len(times)} requested frames, {sum(r["bytes"] for r in manifest["frames"]):,} verified bytes')


if __name__ == '__main__':
    main()
