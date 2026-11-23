"""Shared candidate search and a fixed HR-to-SR direct reference baseline."""
import itertools

import cv2
import numpy as np

from dev_pipeline import luminance, shift, project

HR_FEATURES = ('residual', 'zero_improvement', 'independent_gap', 'magnitude', 'boundary', 'tile_agreement')
QUARTERS = np.arange(-1,1.001,.25)
LR = np.arange(12,116)*4+1
ROI = np.s_[32:480,32:480]


def cubic_coefficients():
    distance = np.abs((np.arange(-7,9)-.5)/4)
    first = 1 + distance**2*(1.5*distance-2.5)
    second = ((-.5*distance+2.5)*distance-4)*distance+2
    kernel = np.where(distance < 1, first, second)
    return kernel/kernel.sum()


KERNEL = cubic_coefficients()


class Projection:
    """Polyphase form of the interior Pillow float bicubic reduction."""
    def __init__(self, moving):
        self.moving = moving
        self.phases = {}

    def __call__(self, vector):
        ix, iy = map(int, np.floor(vector))
        phase = (vector[0]-ix, vector[1]-iy)
        if phase not in self.phases:
            warp = self.moving if phase == (0,0) else shift(self.moving, phase)
            blur = cv2.GaussianBlur(warp, (11,11), 1.2, borderType=cv2.BORDER_REFLECT_101)
            self.phases[phase] = cv2.sepFilter2D(blur, -1, KERNEL, KERNEL, anchor=(7,7))
        return self.phases[phase][np.ix_(LR-iy,LR-ix)]


def standardize(roi):
    centered = roi-roi.mean()
    std = float(roi.std())
    if not np.isfinite(std) or std < 1e-6:
        return None
    return centered/std


def preferred_peaks(ordered):
    peaks = []
    for x,y in ordered:
        if all((x-px)**2+(y-py)**2 >= 4 for px,py in peaks):
            peaks.append((x,y))
            if len(peaks) == 3:
                return peaks
    return peaks


def refined(peaks, radius):
    for x,y in peaks:
        for dx,dy in itertools.product(QUARTERS, repeat=2):
            vector = (float(x+dx),float(y+dy))
            if max(map(abs,vector)) <= radius:
                yield vector


def tie(vector):
    x,y = vector
    return x*x+y*y,x,y


def candidate(moving, lr, reference=False):
    moving = luminance(moving)
    anchor = standardize(luminance(lr)[12:116,12:116])
    invalid = dict(valid=False, vector=[0.,0.], executable=False)
    if anchor is None:
        return dict(invalid, reason='constant_or_nonfinite_anchor')
    if reference:
        def predict(vector):
            return project(shift(moving,vector))[12:116,12:116]
    else:
        predict = Projection(moving)
    errors = {}

    def evaluate(vector):
        predicted = standardize(predict(vector))
        if predicted is None:
            errors[vector] = (np.inf,np.full(16,np.inf))
            return
        residual = (predicted-anchor)**2
        errors[vector] = (float(residual.mean()),residual.reshape(4,26,4,26).mean(axis=(1,3)).ravel())

    def rank(vector):
        return (round(errors[vector][0],10),*tie(vector))

    for vector in itertools.product(range(-8,9), repeat=2):
        evaluate(vector[::-1])
    for vector in refined(preferred_peaks(sorted(errors,key=rank)),8):
        if vector not in errors:
            evaluate(vector)
    best = min(errors,key=rank)
    error = errors[best][0]
    if not np.isfinite(error):
        return dict(invalid, reason='no_finite_projection')
    second = min((v for v in errors if (v[0]-best[0])**2+(v[1]-best[1])**2 >= 4),key=rank)
    vectors = np.array(list(errors))
    tile_vectors = vectors[np.argmin(np.stack([row[1] for row in errors.values()]),axis=0)]
    magnitude = float(np.linalg.norm(best))
    maximum = max(map(abs,best))
    return dict(valid=True,reason=None,vector=list(best),executable=maximum>.25,
                residual=error,zero_improvement=errors[(0,0)][0]-error,
                independent_gap=errors[second][0]-error,independent_vector=list(second),
                magnitude=magnitude,boundary=maximum==8,
                tile_agreement=float(np.mean(np.max(np.abs(tile_vectors-best),axis=1)<=.5)),
                tile_vectors=tile_vectors.tolist(),hypotheses=len(errors))


def direct_candidate(hr, sr):
    """Standardized luminance on the shared 448 px ROI, ±16 px then quarter-pixel refinement."""
    moving = luminance(hr)
    anchor = standardize(luminance(sr)[ROI])
    zero = [0.,0.]
    if anchor is None:
        return zero
    region = moving[16:496,16:496]
    approx = cv2.matchTemplate(region,anchor,cv2.TM_CCOEFF_NORMED)
    ordered = sorted(((16-x,16-y) for y,x in np.ndindex(approx.shape)),
                     key=lambda v: (-float(approx[16-v[1],16-v[0]]),*tie(v)))
    errors = {}
    phases = {(0.,0.):moving}
    for x,y in refined(preferred_peaks(ordered),16):
        ix,iy = int(np.floor(x)),int(np.floor(y))
        phase = (x-ix,y-iy)
        if phase not in phases:
            phases[phase] = shift(moving,phase)
        roi = phases[phase][32-iy:480-iy,32-ix:480-ix]
        predicted = standardize(roi)
        if predicted is not None:
            errors[(x,y)] = round(float(np.mean((predicted-anchor)**2)),10)
    if not errors:
        return zero
    best = min(errors,key=lambda v: (errors[v],*tie(v)))
    return list(best) if max(map(abs,best)) > .25 else zero
