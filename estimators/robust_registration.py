"""Registration objectives for the post-primary local-replacement experiment."""
import itertools

import numpy as np

from dev_pipeline import luminance
from registration import Projection, preferred_peaks, refined, standardize, tie


TILES_PER_AXIS = 8
TILE_WIDTH = 104 // TILES_PER_AXIS
MIN_ANCHOR_STD = 2.0
MIN_TILES = 16


def search(loss):
    errors = {v: loss(v) for v in itertools.product(range(-8, 9), repeat=2)}

    def rank(vector):
        return round(errors[vector], 10), *tie(vector)

    for vector in refined(preferred_peaks(sorted(errors, key=rank)), 8):
        if vector not in errors:
            errors[vector] = loss(vector)
    best = min(errors, key=rank)
    if not np.isfinite(errors[best]):
        return dict(vector=[0, 0], valid=False, executable=False, residual=None, zero_improvement=None)
    return dict(vector=list(best), valid=True, executable=max(map(abs, best)) > .25,
                residual=errors[best], zero_improvement=errors[(0, 0)] - errors[best])


def tiled(array):
    return array.reshape(TILES_PER_AXIS, TILE_WIDTH, TILES_PER_AXIS, TILE_WIDTH).transpose(
        0, 2, 1, 3).reshape(TILES_PER_AXIS ** 2, TILE_WIDTH ** 2)


def tiled_candidates(image, lr):
    projection = Projection(luminance(image))
    anchor = tiled(luminance(lr)[12:116, 12:116])
    selected = anchor.std(axis=1) >= MIN_ANCHOR_STD
    tile_count = int(selected.sum())
    if tile_count < MIN_TILES:
        invalid = dict(vector=[0, 0], valid=False, executable=False, residual=None,
                       zero_improvement=None, texture_tiles=tile_count)
        return dict(tile_mean=invalid, tile_trimmed=invalid)

    anchor = anchor[selected]
    anchor -= anchor.mean(axis=1, keepdims=True)
    anchor /= np.linalg.norm(anchor, axis=1, keepdims=True)
    kept = (tile_count + 1) // 2
    cache = {}

    def losses(vector):
        if vector not in cache:
            tiles = tiled(projection(vector))[selected]
            tiles -= tiles.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(tiles, axis=1)
            similarity = (tiles * anchor).sum(axis=1) / np.maximum(norms, 1e-6)
            error = 1 - np.clip(similarity, -1, 1)
            cache[vector] = float(error.mean()), float(np.partition(error, kept - 1)[:kept].mean())
        return cache[vector]

    mean = search(lambda vector: losses(vector)[0])
    trimmed = search(lambda vector: losses(vector)[1])
    return dict(tile_mean=dict(mean, texture_tiles=tile_count),
                tile_trimmed=dict(trimmed, texture_tiles=tile_count, kept_tiles=kept))


def outside_replacement(image, lr, x, y, size):
    """Oracle-mask diagnostic. Keep fixed support safe for every tested warp."""
    centers = 4 * np.arange(12, 116) + 1.5
    inside_x = (centers >= x - 24) & (centers < x + size + 24)
    inside_y = (centers >= y - 24) & (centers < y + size + 24)
    keep = ~(inside_y[:, None] & inside_x[None, :])
    anchor = standardize(luminance(lr)[12:116, 12:116][keep])
    projection = Projection(luminance(image))
    if anchor is None:
        return dict(vector=[0, 0], valid=False, executable=False, residual=None, zero_improvement=None)

    def loss(vector):
        predicted = standardize(projection(vector)[keep])
        return np.inf if predicted is None else float(np.mean((predicted - anchor) ** 2))

    return dict(search(loss), retained_lr_pixels=int(keep.sum()))
