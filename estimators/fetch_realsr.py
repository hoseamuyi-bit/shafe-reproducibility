"""Fetch the test pairs at their original ZIP member offsets and verify CRCs."""
import concurrent.futures
import io
import random
import re
import struct
import subprocess
import zipfile
import zlib

from PIL import Image

from dev_pipeline import HERE, dump, sha

ROOT = HERE / 'external'
REVISION = '31d8a853131559251f77012e03c4607ce4478c73'
URL = f'https://hf-mirror.com/datasets/Xeno73/RealSR/resolve/{REVISION}/RealSR%28V3%29.zip'
ARCHIVE_SHA = 'b583a6b4812a1edbbc7c0dc732d5beeb219b63dbeed96f2f78ea71d4e90cf2a5'
ARCHIVE_SIZE = 3955899983
CENTRAL_OFFSET = 3955546664
CENTRAL = struct.Struct('<4s6H3I5H2I')


def directory():
    path = ROOT / 'realsr_central.bin'
    if not path.exists():
        fetch_range(CENTRAL_OFFSET, ARCHIVE_SIZE-1, path)
    data = path.read_bytes()
    cursor = 0
    rows = {}
    while data[cursor:cursor+4] == b'PK\x01\x02':
        record = CENTRAL.unpack_from(data,cursor)
        start = cursor + CENTRAL.size
        name = data[start:start+record[10]].decode('utf-8')
        if re.fullmatch(r'(Canon|Nikon)/Test/4/(Canon|Nikon)_\d+_(HR|LR4)\.png',name):
            rows[name] = dict(crc32=record[7],compressed_size=record[8],size=record[9],
                              offset=record[16],method=record[4])
        cursor = start + sum(record[10:13])
    if not rows:
        raise ValueError('No RealSR V3 camera test pairs in the ZIP directory')
    return rows


def fetch_range(start, end, path):
    expected = end-start+1
    result = subprocess.run(['curl.exe','-sS','-f','-L','--connect-timeout','12',
          '--max-time','45','--retry','3','--retry-delay','1','--max-filesize',str(expected),
          '-r',f'{start}-{end}','-o',str(path),'-w','%{http_code} %{size_download}',
          f'{URL}?download=true&part={start}-{end}'],capture_output=True,text=True,check=True)
    if result.stdout.strip() != f'206 {expected}':
        raise ValueError(f'Incorrect range response for offset {start}: {result.stdout}')


def extract(name, record):
    target = ROOT / 'realsr' / name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        data = target.read_bytes()
        if len(data) == record['size'] and zlib.crc32(data) == record['crc32']:
            return target
    chunk = target.with_suffix('.range')
    # The local header's extra field has a 16-bit length. Read a bounded surplus.
    offset = record['offset']
    fetch_range(offset,offset+30+len(name.encode())+65535+record['compressed_size']-1,chunk)
    block = chunk.read_bytes()
    fields = struct.unpack_from('<4s5H3I2H',block)
    nlen, xlen = fields[-2:]
    if fields[0] != b'PK\x03\x04' or fields[3] != record['method'] or block[30:30+nlen].decode() != name:
        raise ValueError(f'ZIP local header mismatch: {name}')
    start = 30+nlen+xlen
    data = block[start:start+record['compressed_size']]
    if record['method'] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data,-15)
    elif record['method'] != zipfile.ZIP_STORED:
        raise ValueError(f'Unexpected compression method for {name}')
    if len(data) != record['size'] or zlib.crc32(data) != record['crc32']:
        raise ValueError(f'CRC or length mismatch: {name}')
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    target.write_bytes(data)
    chunk.unlink()
    return target


def crop_eligible(hr,lr):
    with Image.open(hr) as a, Image.open(lr) as b:
        return all(x == 4*y and y >= 128 for x,y in zip(a.size,b.size))


def main():
    entries = directory()
    selected = []
    available = {}
    rejected = []
    for camera in ('Canon','Nikon'):
        ids = sorted({name.rsplit('/',1)[-1].split('_HR')[0]
                      for name in entries if name.startswith(camera+'/') and name.endswith('_HR.png')})
        rng = random.Random(f'20261122|RealSR-V3|Test|x4|{camera}')
        rng.shuffle(ids)
        available[camera] = len(ids)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0,len(ids),4):
                batch = ids[start:start+4]
                def get_pair(scene):
                    paths = [f'{camera}/Test/4/{scene}_{side}.png' for side in ('HR','LR4')]
                    return scene,paths,[extract(path,entries[path]) for path in paths]
                for scene,names,paths in pool.map(get_pair,batch):
                    if not crop_eligible(*paths):
                        rejected.append(dict(scene=scene,reason='not_exact_x4_or_too_small'))
                        continue
                    selected.append(dict(scene=scene,camera=camera,files=[
                         dict(member=name,path=str(path),sha256=sha(path),**entries[name])
                         for name,path in zip(names,paths)]))
                    if sum(r['camera']==camera for r in selected) == 16:
                        break
                if sum(r['camera']==camera for r in selected) == 16:
                    break
        print(camera,available[camera],sum(r['camera']==camera for r in selected),flush=True)
    if len(selected)!=32:
        raise ValueError('Expected 32 eligible pairs')
    dump(ROOT/'realsr_sources.json',dict(official_repo='https://github.com/csjcai/RealSR',
         mirror=URL,mirror_revision=REVISION,mirror_reported_sha256=ARCHIVE_SHA,
         archive_size=ARCHIVE_SIZE,central_directory_sha256=sha(ROOT/'realsr_central.bin'),
         available=available,rejected=rejected,selected=selected))


if __name__ == '__main__':
    main()
