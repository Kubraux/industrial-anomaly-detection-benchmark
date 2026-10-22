import numpy as np
from scipy import ndimage

def pro_metric(scores, gt):
    weights = np.zeros(gt.shape, np.float64)
    regions = 0
    for i, g in enumerate(gt):
        lab, n = ndimage.label(g, np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        lut = np.zeros(n + 1)
        if n:
            lut[1:] = 1 / sizes[1:]
        weights[i] = lut[lab]
        regions += n
    assert regions and (~gt).any()
    weights /= regions
    flat = scores.ravel()
    order = np.argsort(-flat, kind='stable')
    s = flat[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    fpr = np.r_[0.0, np.cumsum((~gt).ravel()[order], dtype=np.int64)[ends] / int((~gt).sum())]
    pro = np.r_[0.0, np.cumsum(weights.ravel()[order])[ends]]
    last = np.searchsorted(fpr, 0.3, side='right')
    x = fpr[:last]
    y = pro[:last]
    if x[-1] < 0.3:
        x = np.r_[x, 0.3]
        y = np.r_[y, np.interp(0.3, fpr[last - 1:last + 1], pro[last - 1:last + 1])]
    return (float(np.trapz(y, x) / 0.3), regions, (fpr, pro))

def pro_reference(scores, gt):
    regs = []
    for i, g in enumerate(gt):
        lab, n = ndimage.label(g, np.ones((3, 3)))
        regs.extend(((i, lab == j) for j in range(1, n + 1)))
    points = [(0.0, 0.0)]
    for t in sorted(np.unique(scores), reverse=True):
        pred = scores >= t
        points.append((float(pred[~gt].mean()), float(np.mean([pred[i][r].mean() for i, r in regs]))))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x1 == x0 or x0 >= 0.3:
            continue
        end = min(x1, 0.3)
        ye = y0 + (y1 - y0) * (end - x0) / (x1 - x0)
        area += (end - x0) * (y0 + ye) / 2
    return area / 0.3