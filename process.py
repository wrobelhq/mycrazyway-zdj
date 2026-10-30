import os, sys, urllib.request, io
import numpy as np
from PIL import Image
from scipy import ndimage

P = 'https://cdn.shopify.com/s/files/1/0683/8560/3641/files/'
os.makedirs('engwe', exist_ok=True)
names = [l.strip() for l in open('lista.txt', encoding='utf-8') if l.strip()]
ok = err = 0
for f in names:
    out = os.path.join('engwe', os.path.splitext(f)[0] + ('_jpg' if f == '10.jpg' else '') + '.jpg')
    if os.path.exists(out):
        continue
    try:
        req = urllib.request.Request(P + f + '?width=1200', headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req, timeout=60).read()
        im = Image.open(io.BytesIO(data))
        if im.mode in ('RGBA', 'LA', 'P'):
            im = im.convert('RGBA')
            bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg
        im = im.convert('RGB')
        a = np.asarray(im).astype(np.int16)
        mn = a.min(axis=2); mx = a.max(axis=2)
        cand = (mn >= 222) & ((mx - mn) <= 12)
        lab, n = ndimage.label(cand)
        border = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
        border = border[border > 0]
        bgmask = np.isin(lab, border)
        out_a = a.copy()
        out_a[bgmask] = 255
        # feather: pixels next to background that are light get blended to white
        edge = ndimage.binary_dilation(bgmask) & ~bgmask
        light = edge & (a.mean(axis=2) > 200)
        out_a[light] = (out_a[light] + 255) // 2
        Image.fromarray(out_a.clip(0, 255).astype(np.uint8)).save(out, 'JPEG', quality=92)
        ok += 1
    except Exception as e:
        print('ERR', f, e); err += 1
print('ok', ok, 'err', err)
