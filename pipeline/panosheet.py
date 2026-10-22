import sys, os
import numpy as np
from PIL import Image, ImageDraw
res = sys.argv[1]; out = sys.argv[2]; n = int(sys.argv[3])
rows = [l.split(',') for l in open(os.path.join(res, os.path.basename(os.path.dirname(res + '/')) and '') or '', 'r')] if False else None
m = os.path.basename(os.path.dirname(res.rstrip('\\/')))
lines = [l.strip().split(',') for l in open(os.path.join(res, m + '.imglist')) if l.strip() and not l.startswith('#')]
a0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0; a1 = int(sys.argv[5]) if len(sys.argv) > 5 else len(lines) - 1
idx = np.linspace(a0, a1, n).astype(int)
tiles = []
for i in idx:
    t, path, cam, x, y, z = lines[i][:6]
    im = Image.open(os.path.join(res, path)); im.draft('RGB', (im.width // 8, im.height // 8)); im = im.convert('RGB').resize((776, 388))
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 330, 22], fill=(0, 0, 0)); d.text((4, 4), f"#{i} t={float(t):.0f} x={float(x):.0f} y={float(y):.0f}", fill=(255, 255, 0))
    tiles.append(im)
cols = 3; R = (len(tiles) + cols - 1) // cols
sheet = Image.new('RGB', (776 * cols, 388 * R))
for k, im in enumerate(tiles): sheet.paste(im, ((k % cols) * 776, (k // cols) * 388))
sheet.save(out, quality=80); print(len(lines))
