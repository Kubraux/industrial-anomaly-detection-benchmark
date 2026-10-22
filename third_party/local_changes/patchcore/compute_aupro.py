from pathlib import Path
import argparse
import csv

import numpy as np
import torch
from sklearn.metrics import auc
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader

import patchcore.common
import patchcore.patchcore
from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit


CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def compute_aupro(masks, anomaly_maps, fpr_limit=0.30, num_thresholds=200):
    masks = np.asarray(masks)

    # PatchCore mask yapısı: [N, 1, H, W]
    if masks.ndim == 4:
        masks = masks[:, 0]

    masks = masks > 0.5
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float64)

    if masks.shape != anomaly_maps.shape:
        raise ValueError(
            f"Shape mismatch: masks={masks.shape}, maps={anomaly_maps.shape}"
        )

    # Ground-truth'taki her connected defect region'ı bir kere çıkar.
    regions_per_image = []

    for mask in masks:
        labelled = label(mask)
        regions = [
            region.coords
            for region in regionprops(labelled)
        ]
        regions_per_image.append(regions)

    normal_pixels = ~masks
    n_normal_pixels = normal_pixels.sum()

    min_score = float(anomaly_maps.min())
    max_score = float(anomaly_maps.max())

    if max_score <= min_score:
        return 0.0

    # Yüksek threshold -> düşük threshold
    thresholds = np.linspace(
        max_score + 1e-12,
        min_score - 1e-12,
        num_thresholds,
    )

    fprs = []
    pros = []

    for threshold in thresholds:
        predicted = anomaly_maps > threshold

        # False Positive Rate:
        # gerçek normal olduğu halde anomaly denilen pikseller
        fp = np.logical_and(predicted, normal_pixels).sum()
        fpr = fp / n_normal_pixels

        region_overlaps = []

        for prediction, regions in zip(
            predicted, regions_per_image
        ):
            for coords in regions:
                detected = prediction[
                    coords[:, 0],
                    coords[:, 1]
                ].mean()

                region_overlaps.append(detected)

        pro = (
            float(np.mean(region_overlaps))
            if region_overlaps
            else 0.0
        )

        fprs.append(float(fpr))
        pros.append(pro)

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)

    # FPR monotonik sırada olsun.
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    # FPR = 0.30 noktasını interpolasyonla ekle.
    pro_at_limit = np.interp(
        fpr_limit,
        fprs,
        pros,
    )

    keep = fprs < fpr_limit

    x = np.concatenate(
        [fprs[keep], [fpr_limit]]
    )
    y = np.concatenate(
        [pros[keep], [pro_at_limit]]
    )

    # Alanı 0.30'a bölerek sonucu 0-1 aralığına getiriyoruz.
    aupro = auc(x, y) / fpr_limit

    return float(aupro)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models_root",
        required=True,
        help="15 mvtec_* model klasörünün bulunduğu models klasörü",
    )

    parser.add_argument(
        "--data_root",
        default="../archive",
    )

    parser.add_argument(
        "--output",
        default="patchcore_aupro_results.csv",
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    results = []

    for category in CATEGORIES:
        print()
        print("=" * 60)
        print(f"PRO hesaplanıyor: {category}")
        print("=" * 60)

        model_path = Path(args.models_root) / f"mvtec_{category}"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {model_path}"
            )

        dataset = MVTecDataset(
            args.data_root,
            classname=category,
            resize=256,
            imagesize=224,
            split=DatasetSplit.TEST,
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        nn_method = patchcore.common.FaissNN(
            False,  # GPU kullanma
            1,
        )

        model = patchcore.patchcore.PatchCore(device)

        model.load_from_path(
            load_path=str(model_path),
            device=device,
            nn_method=nn_method,
        )

        _, segmentations, _, masks_gt = model.predict(loader)

        segmentations = np.asarray(segmentations)

        # Resmi evaluation kodundaki global min-max normalization.
        seg_min = segmentations.min()
        seg_max = segmentations.max()

        segmentations = (
            (segmentations - seg_min)
            / (seg_max - seg_min + 1e-12)
        )

        score = compute_aupro(
            masks_gt,
            segmentations,
            fpr_limit=0.30,
            num_thresholds=200,
        )

        results.append((category, score))

        print(f"{category}: AUPRO@0.30 = {score:.4f}")

    mean_score = float(
        np.mean([x[1] for x in results])
    )

    print()
    print("=" * 60)
    print(f"MEAN AUPRO@0.30 = {mean_score:.4f}")
    print("=" * 60)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["category", "aupro_0.30"]
        )

        for category, score in results:
            writer.writerow(
                [category, score]
            )

        writer.writerow(
            ["Mean", mean_score]
        )

    print(f"Sonuçlar kaydedildi: {args.output}")


if __name__ == "__main__":
    main()