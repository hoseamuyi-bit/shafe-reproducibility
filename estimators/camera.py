"""Prepare aligned-center crops from the independently selected RealSR test pairs."""
import random

from dev_pipeline import CONFIG,HERE,dump,pixel_sha,read,save,sha
from experiment import DIRECTIONS,load
from freeze import LOCK,verify

ROOT = HERE/'external'


def run():
    verify()
    sources = ROOT/'realsr_sources.json'
    selected = load(sources)['selected']
    assert len({row['scene'] for row in selected})==32
    rows,blind = [],[]
    for camera in ('Canon','Nikon'):
        pairs = sorted((r for r in selected if r['camera']==camera),key=lambda r:r['scene'])
        if len(pairs)!=16:
            raise ValueError('Expected 16 test scenes per camera')
        rng = random.Random(f'20261122|external|RealSR-V3|{camera}|injection-v2')
        rng.shuffle(pairs)
        for index,pair in enumerate(pairs):
            scene = f"RealSR_{pair['scene']}"
            hr_record,lr_record = pair['files']
            hp,lp = hr_record['path'],lr_record['path']
            if (sha(hp),sha(lp))!=(hr_record['sha256'],lr_record['sha256']):
                raise ValueError(f'Original RealSR pairs changed: {scene}')
            hr,lr = read(hp),read(lp)
            h,w,_ = lr.shape
            assert hr.shape==(4*h,4*w,3) and min(h,w)>=128
            x,y = (w-128)//2,(h-128)//2
            lr_xyxy = [x,y,x+128,y+128]
            hr_xyxy = [4*v for v in lr_xyxy]
            lr_crop = lr[y:y+128,x:x+128]
            hr_crop = hr[4*y:4*y+512,4*x:4*x+512]
            lr_out = ROOT/'lr'/f'{scene}__camera.png'
            hr_out = ROOT/'inputs'/scene/'hr.png'
            sr_out = ROOT/'vosr/images'/lr_out.name
            save(lr_out,lr_crop)
            save(hr_out,hr_crop)
            row = dict(scene=scene,group=scene,dataset=camera,degradation='camera',
                source=hp,source_sha256=hr_record['sha256'],camera_lr_source=lp,
                camera_lr_source_sha256=lr_record['sha256'],lr_crop_xyxy=lr_xyxy,hr_crop_xyxy=hr_xyxy,
                magnitude=(.5,2,6)[index%3],hr_direction=list(rng.choice(DIRECTIONS)),
                sr_direction=list(rng.choice(DIRECTIONS)),hr=str(hr_out),lr=str(lr_out),sr=str(sr_out),
                hr_sha256=sha(hr_out),lr_sha256=sha(lr_out),lr_pixel_sha256=pixel_sha(lr_crop))
            rows.append(row)
            blind.append(dict(scene=scene,camera=camera,hr=str(hr_out),lr=str(lr_out),sr=str(sr_out)))
    dump(ROOT/'batch_inputs.json',dict(split='external',design_sha256=sha(sources),
        formal_lock_sha256=sha(LOCK),config=CONFIG,records=rows))
    dump(ROOT/'landmark_inputs.json',dict(scenes=blind))
    print('external: 32 center-cropped camera pairs',flush=True)


if __name__=='__main__':
    run()
