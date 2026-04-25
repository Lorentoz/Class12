"""
generate_shelf_images.py
Generates 900 synthetic 64x64 grayscale shelf images (300 per class).
Classes: normal (0), damaged (1), overloaded (2).
Run once to produce shelf_images.npz.
"""

import numpy as np

IMG_SIZE         = 64
SHELF_THICKNESS  = 5
BOX_REGION_TOP   = 4
BG_BRIGHTNESS    = 0.12
SHELF_BRIGHTNESS = 0.75


def _random_shelf_top(rng):
    center = IMG_SIZE // 2
    return center + rng.integers(-4, 5)


def _draw_shelf(img, shelf_top, rng):
    shelf_bottom = shelf_top + SHELF_THICKNESS
    img[shelf_top:shelf_bottom, :] = SHELF_BRIGHTNESS
    n_dividers = rng.integers(3, 7)
    for _ in range(n_dividers):
        x = rng.integers(5, IMG_SIZE - 5)
        img[shelf_top:shelf_bottom, max(0, x-1):x+1] = SHELF_BRIGHTNESS * 0.6


def _draw_boxes(img, shelf_top, n_boxes, max_height, rng):
    box_bottom = shelf_top
    x = rng.integers(2, 8)
    for _ in range(n_boxes):
        w = rng.integers(4, 12)
        h = rng.integers(4, max_height)
        box_top = box_bottom - h
        if box_top < BOX_REGION_TOP:
            break
        x_end = min(x + w, IMG_SIZE)
        actual_w = x_end - x
        if actual_w <= 0:
            break
        brightness = rng.uniform(0.3, 0.55)
        img[box_top:box_bottom, x:x_end] = np.clip(
            brightness + rng.normal(0, 0.03, (h, actual_w)), 0.15, 0.65
        )
        x += w + rng.integers(1, 4)
        if x >= IMG_SIZE - 6:
            break


def _draw_crack(img, shelf_top, rng):
    shelf_bottom = shelf_top + SHELF_THICKNESS
    x_start = rng.integers(5, IMG_SIZE - 15)
    length  = rng.integers(8, 20)
    angle   = rng.uniform(-0.5, 0.5)
    for i in range(length):
        x = int(x_start + i)
        y = int(shelf_top + 1 + i * angle)
        if 0 <= x < IMG_SIZE and shelf_top <= y < shelf_bottom:
            img[y, x] = 0.05


def _box_area_fraction(img, shelf_top):
    region = img[BOX_REGION_TOP:shelf_top, :]
    if region.size == 0:
        return 0.0
    return (region > BG_BRIGHTNESS + 0.1).mean()


def _add_noise(img, rng):
    img = img + rng.uniform(-0.04, 0.04)
    img = img + rng.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1)


def generate_normal(rng):
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS, dtype=np.float32)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 5), max_height=15, rng=rng)
    attempts = 0
    while _box_area_fraction(img, shelf_top) > 0.25 and attempts < 10:
        img[BOX_REGION_TOP:shelf_top, :] = BG_BRIGHTNESS
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=12, rng=rng)
        attempts += 1
    return _add_noise(img, rng)


def generate_damaged(rng):
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS, dtype=np.float32)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 5), max_height=15, rng=rng)
    attempts = 0
    while _box_area_fraction(img, shelf_top) > 0.25 and attempts < 10:
        img[BOX_REGION_TOP:shelf_top, :] = BG_BRIGHTNESS
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=12, rng=rng)
        attempts += 1
    n_cracks = rng.integers(1, 3)
    for _ in range(n_cracks):
        _draw_crack(img, shelf_top, rng)
    return _add_noise(img, rng)


def generate_overloaded(rng):
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS, dtype=np.float32)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(6, 10), max_height=35, rng=rng)
    attempts = 0
    while _box_area_fraction(img, shelf_top) < 0.45 and attempts < 20:
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=30, rng=rng)
        attempts += 1
    return _add_noise(img, rng)


def generate_dataset(n_per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    generators  = [generate_normal, generate_damaged, generate_overloaded]
    class_names = ["normal", "damaged", "overloaded"]
    images, labels = [], []
    for class_idx, gen in enumerate(generators):
        for _ in range(n_per_class):
            images.append(gen(rng))
            labels.append(class_idx)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    idx    = rng.permutation(len(images))
    return images[idx], labels[idx], class_names


if __name__ == "__main__":
    images, labels, class_names = generate_dataset(n_per_class=300, seed=42)
    print(f"Images: {images.shape}  (min={images.min():.2f}, max={images.max():.2f})")
    print(f"Labels: {labels.shape},  classes: {class_names}")
    print(f"Class distribution: {[int((labels==i).sum()) for i in range(3)]}")
    np.savez("shelf_images.npz", images=images, labels=labels, class_names=class_names)
    print("Saved shelf_images.npz")
