import argparse
import os
import time
from PIL import ImageGrab
import constants

# Always load the latest coordinates
COORDS = constants.load_coordinates()


def absolute_coords(region_name, ref=(0, 0)):
    """Return absolute coordinates for a region from constants."""
    info = COORDS.get(region_name)
    if not info:
        return None
    x1, y1, x2, y2 = info['coord_rel']
    rx, ry = ref
    return (x1 + rx, y1 + ry, x2 + rx, y2 + ry)


def parse_bbox(text):
    """Parse bbox string 'x1,y1,x2,y2' into a tuple of ints."""
    parts = [int(p) for p in text.split(',')]
    if len(parts) != 4:
        raise ValueError('bbox must have four integers')
    return tuple(parts)


def test_card(img, threshold=240, min_white=10):
    """Return True if at least ``min_white`` pixels near bottom-right are white."""
    w, h = img.size
    zone = img.crop((max(0, w - 5), max(0, h - 5), w, h))
    white_pixels = 0
    for px in zone.getdata():
        if all(c >= threshold for c in px[:3]):
            white_pixels += 1
    return white_pixels >= min_white


def capture(number_bbox, symbol_bbox):
    num_img = ImageGrab.grab(bbox=number_bbox)
    sym_img = ImageGrab.grab(bbox=symbol_bbox)
    if not test_card(num_img):
        print('Card test failed: not enough white pixels.')
        return

    value = input('Enter card value: ').strip()
    suit = input('Enter card suit: ').strip()

    num_dir = os.path.join('screen', 'debug', 'Carte', value)
    sym_dir = os.path.join('screen', 'symbole', suit)
    os.makedirs(num_dir, exist_ok=True)
    os.makedirs(sym_dir, exist_ok=True)
    ts = int(time.time())
    num_img.save(os.path.join(num_dir, f'{ts}.png'))
    sym_img.save(os.path.join(sym_dir, f'{ts}.png'))
    print('Images saved.')


def main():
    parser = argparse.ArgumentParser(description='Capture card images from the screen.')
    parser.add_argument('--number', help='Region name or x1,y1,x2,y2 for the card value')
    parser.add_argument('--symbol', help='Region name or x1,y1,x2,y2 for the card suit')
    parser.add_argument('--ref', help='Reference point x,y added to coordinates from constants')
    args = parser.parse_args()

    ref = (0, 0)
    if args.ref:
        rx, ry = [int(v) for v in args.ref.split(',')]
        ref = (rx, ry)

    if args.number in COORDS:
        number_bbox = absolute_coords(args.number, ref)
    elif args.number:
        number_bbox = parse_bbox(args.number)
    else:
        number_bbox = parse_bbox(input('Number bbox (x1,y1,x2,y2): '))

    if args.symbol in COORDS:
        symbol_bbox = absolute_coords(args.symbol, ref)
    elif args.symbol:
        symbol_bbox = parse_bbox(args.symbol)
    else:
        symbol_bbox = parse_bbox(input('Symbol bbox (x1,y1,x2,y2): '))

    capture(number_bbox, symbol_bbox)


if __name__ == '__main__':
    main()
