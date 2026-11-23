"""Compare all source scenes for overlap, without looking at method outcomes."""
import itertools

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dev_pipeline import HERE, dump
from experiment import load


def review_candidates():
    cv2.setNumThreads(4)
    rows = load(HERE/'data/manifest_provisional.json')['scenes']
    descriptors, points, previews = [], [], []
    sift = cv2.SIFT_create(500)
    for row in rows:
        with Image.open(row['source']) as image:
            reduced = ImageOps.contain(image.convert('RGB'),(600,600))
        kp, desc = sift.detectAndCompute(np.array(reduced.convert('L')),None)
        descriptors.append(desc)
        points.append(np.array([key.pt for key in kp],dtype='float32'))
        previews.append(ImageOps.contain(reduced,(142,108)))
    # Each descriptor retrieves 12 distinct near neighbors. Count candidate
    # scene pairs, then re-match those pairs independently with a ratio test.
    ids = np.repeat(np.arange(len(rows)),[len(d) for d in descriptors])
    database = np.vstack(descriptors)
    cv2.setRNGSeed(20261122)
    index = cv2.flann_Index(database,dict(algorithm=1,trees=5))
    votes = {}
    for i,desc in enumerate(descriptors):
        indices, _ = index.knnSearch(desc,12,params=dict(checks=64))
        counts = np.bincount(ids[indices].ravel(),minlength=len(rows))
        for j in np.flatnonzero(counts >= 50):
            if j != i:
                key = tuple(sorted((i,int(j))))
                votes[key] = max(int(counts[j]),votes.get(key,0))
    found = []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for (i,j),votes_ij in sorted(votes.items()):
        # Preserve mutual descriptors to reduce false matches in repetitive buildings.
        forwards = matcher.knnMatch(descriptors[i],descriptors[j],k=2)
        backwards = matcher.knnMatch(descriptors[j],descriptors[i],k=2)
        forward = {a.queryIdx:a.trainIdx for a,b in forwards if a.distance < .8*b.distance}
        backward = {a.queryIdx:a.trainIdx for a,b in backwards if a.distance < .8*b.distance}
        pairs = [(x,y) for x,y in forward.items() if backward.get(y) == x]
        if len(pairs) < 10:
            continue
        qi,qj = np.array(pairs).T
        _, keep = cv2.findHomography(points[i][qi],points[j][qj],cv2.RANSAC,4.)
        if keep is None or keep.sum() < 10:
            continue
        found.append(dict(first=rows[i]['id'],second=rows[j]['id'],first_idx=i,second_idx=j,
                          descriptor_votes=votes_ij,mutual_matches=len(pairs),
                          geometric_inliers=int(keep.sum())))
    found.sort(key=lambda r:-r['geometric_inliers'])
    dump(HERE/'data/scene_overlap_candidates.json',
         dict(screened=len(rows),candidate_pairs_independently_matched=len(votes),pairs=found))
    tiles = [(f"{i+1} {r['id'].replace('_SRF_4_HR','')}",p) for i,(r,p) in enumerate(zip(rows,previews))]
    for page, start in enumerate(range(0,len(tiles),50),1):
        canvas = Image.new('RGB',(1500,660),'#f1f1f1')
        draw = ImageDraw.Draw(canvas)
        for idx,(name,preview) in enumerate(tiles[start:start+50]):
            x,y = idx%10*150,idx//10*132
            canvas.paste(preview,(x+4,y+3))
            draw.text((x+4,y+112),name,fill='#101010')
        canvas.save(HERE/f'data/scene_contact_{page}.jpg',quality=89)
    if found:
        canvas = Image.new('RGB',(720,145*((len(found)+1)//2)),'#eeeeee')
        draw = ImageDraw.Draw(canvas)
        for idx,pair in enumerate(found):
            x,y = idx%2*360,idx//2*145
            for col,key in enumerate(('first','second')):
                canvas.paste(previews[pair[key+'_idx']],(x+col*155,y))
            draw.text((x,y+111),f"{pair['first']} | {pair['second']}",fill='#101010')
            draw.text((x,y+125),f"{pair['geometric_inliers']} inliers",fill='#101010')
        canvas.save(HERE/'data/scene_candidates.jpg',quality=90)
    print(f'{len(rows)} scenes, {len(votes)} matched pairs, {len(found)} flagged',flush=True)


if __name__ == '__main__':
    review_candidates()
