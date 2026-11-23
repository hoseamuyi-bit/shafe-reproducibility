"""Render a deterministic sample of the COCO registration failures."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coco_confirm import FOLDER, INPUTS
from holdout_run import atomic_dump, images_for, load, variants_for
from verify_holdout import event


def display(array, size):
    pixels = np.rint(np.clip(array, 0, 255)).astype(np.uint8)
    return Image.fromarray(pixels).resize((size, size), Image.Resampling.BICUBIC)


def make_examples(output):
    geometry = sorted(load(FOLDER / 'geometry.json')['rows'], key=lambda r: r['scene'])
    scores = {(r['scene'], r['condition']): r for r in load(FOLDER / 'scores.json')['rows']}
    records = {r['scene']: r for r in load(INPUTS)['records']}
    methods = ('simple_lr', 'tile_mean', 'tile_trimmed', 'median_lk', 'oracle')
    labels = {'simple_lr': 'Global LR', 'tile_mean': 'Mean tiles', 'tile_trimmed': 'Trimmed tiles',
              'median_lk': 'Median LK', 'oracle': 'Known inverse'}
    examples = [
        ('Global LR moves an already positioned reference', 'replacement_256',
         lambda r: event(r, 'simple_lr', 'false') and not event(r, 'median_lk', 'false') and r['tracks'] >= 8),
        ('Global LR misses the added 2 px translation', 'shift2_replacement_256',
         lambda r: not event(r, 'simple_lr', 'correct') and event(r, 'median_lk', 'correct')),
        ('Median LK holds because too few feature tracks survive', 'shift2_replacement_256',
         lambda r: r['tracks'] < 8 and not event(r, 'median_lk', 'correct')),
        ('Trimmed tiles miss a clean 0.5 px translation', 'hr_global',
         lambda r: r['magnitude'] == .5 and not event(r, 'tile_trimmed', 'correct')
         and event(r, 'median_lk', 'correct')),
    ]
    size, gap = 226, 14
    sheet = Image.new('RGB', (5*gap+4*size, 4*345+gap), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    title_font, text_font = ImageFont.load_default(size=17), ImageFont.load_default(size=14)
    metadata = []
    for i, (label, condition, predicate) in enumerate(examples):
        row = next(r for r in geometry if r['condition'] == condition and predicate(r))
        scene = row['scene']
        lr, hr, sr, full = images_for(records[scene])
        _, moving, fixed, translation, _, patch = next(
            v for v in variants_for(records[scene], hr, sr, full) if v[0] == condition)
        images = [display(array, size) for array in (lr, hr, moving, fixed)]
        if patch:
            x, y = np.array(patch['target_xy']) + translation
            area = np.array([x, y, x+patch['size'], y+patch['size']]) * size / 512
            ImageDraw.Draw(images[2]).rectangle(tuple(area), outline='#f2d34b', width=2)
        top = gap+i*345
        sampled = scores[scene, condition]
        draw.text((gap, top), f'{label}  |  {scene}', fill='#161c26', font=title_font)
        for j, (image, caption) in enumerate(zip(images, ('LR (display enlarged)', 'Original HR',
                                                         'Perturbed HR', 'Fixed VOSR output'), strict=True)):
            left = gap+j*(size+gap)
            draw.text((left, top+26), caption, fill='#353f50', font=text_font)
            sheet.paste(image, (left, top+46))
        actions = []
        for method in methods:
            x, y = sampled['actions'][method]
            actions.append(f'{labels[method]} ({x:g}, {y:g})')
        added = ', '.join(f'{v:g}' for v in translation)
        draw.text((gap, top+282), f'Injected HR ({added}) px  |  {row["tracks"]} accepted LK tracks',
                  fill='#353f50', font=text_font)
        draw.text((gap, top+306), 'Actions in HR pixels  |  ' + '   '.join(actions),
                  fill='#161c26', font=text_font)
        metadata.append(dict(description=label, scene=scene, condition=condition,
                             source_url=records[scene]['source_url'], source_license=records[scene]['source_license'],
                             tracks=row['tracks'], hr_added=row['hr_added'], patch=patch,
                             actions=sampled['actions'], scores=sampled['scores']))
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / 'COCO_failure_examples.png')
    atomic_dump(output / 'COCO_failure_examples.json', dict(
        selection='Lexicographically first matching scene for each listed criterion; selected for illustration.',
        cases=metadata))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    make_examples(parser.parse_args().output)
