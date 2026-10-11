"""Per-detection match tables: COCO-style AP that can be recomputed under image resampling.

The cyclist study's decision rule (``docs/riders/preregistration.md``, section 5, criterion 3) asks
for a paired bootstrap over validation images. COCOeval cannot be rerun 10,000 times per model, so
the matching is done **once** per model and stored; an AP for any weighting of the images (a
bootstrap resample is integer weights) is then a few cumulative sums.

``build_match_table`` reproduces COCOeval's matching for the ``all`` area range with
``maxDets`` as given: detections of one image and class sorted by score, each taking the best
still-unmatched ground truth at each IoU threshold 0.50, 0.55, ... 0.95, non-ignored ground truths
preferred, a detection matched to a crowd region ignored (neither true nor false positive).
``average_precision`` follows COCOeval's accumulation: monotone precision envelope sampled at 101
recall thresholds. The tests compare both with ``pycocotools`` on random data; the whole path is
checked again against each model's published AP when the tables are built.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
EPS = np.spacing(1)


def _iou(  # pylint: disable=too-many-locals
    detections: npt.NDArray[np.float64],
    truths: npt.NDArray[np.float64],
    crowd: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """IoU of xywh ``detections`` (D, 4) and ``truths`` (G, 4); crowd columns use det area."""
    dx1, dy1 = detections[:, 0:1], detections[:, 1:2]
    dx2, dy2 = dx1 + detections[:, 2:3], dy1 + detections[:, 3:4]
    gx1, gy1 = truths[None, :, 0], truths[None, :, 1]
    gx2, gy2 = gx1 + truths[None, :, 2], gy1 + truths[None, :, 3]
    width = np.clip(np.minimum(dx2, gx2) - np.maximum(dx1, gx1), 0.0, None)
    height = np.clip(np.minimum(dy2, gy2) - np.maximum(dy1, gy1), 0.0, None)
    inter = width * height
    area_d = detections[:, 2:3] * detections[:, 3:4]
    area_g = (truths[:, 2] * truths[:, 3])[None, :]
    union = np.where(crowd[None, :], area_d, area_d + area_g - inter)
    return np.asarray(inter / np.maximum(union, EPS))


def _match_image(  # pylint: disable=too-many-locals
    boxes: npt.NDArray[np.float64],
    truths: npt.NDArray[np.float64],
    crowd: npt.NDArray[np.bool_],
) -> Any:
    """(matched-to-real, ignored) booleans of shape (T, D) for one image and class.

    ``boxes`` are sorted by descending score. Ground truths are ordered non-crowd first.
    """
    thresholds = len(IOU_THRESHOLDS)
    detections = len(boxes)
    matched = np.zeros((thresholds, detections), dtype=bool)
    ignored = np.zeros((thresholds, detections), dtype=bool)
    if detections == 0 or len(truths) == 0:
        return matched, ignored
    order = np.argsort(crowd, kind="mergesort")  # non-crowd first
    truths, crowd = truths[order], crowd[order]
    ious = _iou(boxes, truths, crowd)
    for t_index, threshold in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(len(truths), dtype=bool)
        for d_index in range(detections):
            best, chosen = min(threshold, 1.0 - 1e-10), -1
            for g_index in range(len(truths)):
                if taken[g_index] and not crowd[g_index]:
                    continue
                if chosen > -1 and not crowd[chosen] and crowd[g_index]:
                    break
                if ious[d_index, g_index] < best:
                    continue
                best, chosen = ious[d_index, g_index], g_index
            if chosen == -1:
                continue
            if crowd[chosen]:
                ignored[t_index, d_index] = True
            else:
                matched[t_index, d_index] = True
                taken[chosen] = True
    return matched, ignored


def build_match_table(  # pylint: disable=too-many-locals
    coco: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
    max_dets: int = 100,
) -> Dict[str, Any]:
    """The match table of ``detections`` against the COCO-format ground truth ``coco``.

    Returns ``{"images": [image ids], "classes": {category_id: {"score", "image", "matched",
    "ignored", "gt_count"}}}`` with detections sorted by descending score across images.
    """
    image_ids = [image["id"] for image in coco["images"]]
    position = {image_id: k for k, image_id in enumerate(image_ids)}
    categories = sorted(c["id"] for c in coco["categories"])
    truths: Dict[Any, List[Dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        truths.setdefault((annotation["image_id"], annotation["category_id"]), []).append(
            annotation
        )
    found: Dict[Any, List[Dict[str, Any]]] = {}
    for detection in detections:
        found.setdefault((detection["image_id"], detection["category_id"]), []).append(detection)
    classes: Dict[int, Dict[str, npt.NDArray[Any]]] = {}
    for category in categories:
        scores, images, matched, ignored = [], [], [], []
        gt_count = np.zeros(len(image_ids), dtype=np.int64)
        for image_id in image_ids:
            gts = truths.get((image_id, category), [])
            crowd = np.array([bool(g.get("iscrowd", 0)) for g in gts], dtype=bool)
            gt_count[position[image_id]] = int((~crowd).sum())
            dets = sorted(found.get((image_id, category), []), key=lambda d: -d["score"])
            dets = dets[:max_dets]
            if not dets:
                continue
            boxes = np.array([d["bbox"] for d in dets], dtype=np.float64)
            boxes_gt = np.array([g["bbox"] for g in gts], dtype=np.float64).reshape(-1, 4)
            m, i = _match_image(boxes, boxes_gt, crowd)
            scores.append([d["score"] for d in dets])
            images.append([position[image_id]] * len(dets))
            matched.append(m)
            ignored.append(i)
        if scores:
            score = np.concatenate([np.array(s) for s in scores])
            order = np.argsort(-score, kind="mergesort")
            classes[category] = {
                "score": score[order],
                "image": np.concatenate([np.array(x) for x in images])[order],
                "matched": np.concatenate(matched, axis=1)[:, order],
                "ignored": np.concatenate(ignored, axis=1)[:, order],
                "gt_count": gt_count,
            }
        else:
            empty = np.zeros((len(IOU_THRESHOLDS), 0), dtype=bool)
            classes[category] = {
                "score": np.zeros(0),
                "image": np.zeros(0, dtype=np.int64),
                "matched": empty,
                "ignored": empty,
                "gt_count": gt_count,
            }
    return {"images": image_ids, "classes": classes}


def average_precision(  # pylint: disable=too-many-locals
    table_class: Dict[str, npt.NDArray[Any]], weights: Optional[npt.NDArray[np.float64]] = None
) -> float:
    """AP@[.5:.95] of one class, optionally with per-image ``weights`` (bootstrap counts).

    Returns ``nan`` when the (weighted) class has no ground truth, as COCOeval's -1.
    """
    image = table_class["image"]
    count = table_class["gt_count"].astype(np.float64)
    w_img = np.ones(len(count)) if weights is None else weights
    total = float((w_img * count).sum())
    if total <= 0:
        return float("nan")
    w_det = w_img[image]
    values = []
    for t_index in range(len(IOU_THRESHOLDS)):
        keep = ~table_class["ignored"][t_index]
        tp = table_class["matched"][t_index][keep] * w_det[keep]
        fp = (~table_class["matched"][t_index][keep]) * w_det[keep]
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        recall = ctp / total
        precision = ctp / (ctp + cfp + EPS)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        index = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
        sampled = np.zeros(len(RECALL_THRESHOLDS))
        valid = index < len(precision)
        sampled[valid] = precision[index[valid]]
        values.append(sampled.mean())
    return float(np.mean(values))


def save_table(path: Any, table: Dict[str, Any]) -> None:
    """Write ``table`` as a compressed ``.npz``."""
    arrays: Dict[str, Any] = {"images": np.array(table["images"], dtype=np.int64)}
    for category, columns in table["classes"].items():
        for name, array in columns.items():
            arrays[f"c{category}_{name}"] = array
    np.savez_compressed(path, **arrays)


def load_table(path: Any) -> Dict[str, Any]:
    """Read a table written by ``save_table``."""
    data = np.load(path)
    classes: Dict[int, Dict[str, Any]] = {}
    for key in data.files:
        if key == "images":
            continue
        prefix, name = key.split("_", 1)
        classes.setdefault(int(prefix[1:]), {})[name] = data[key]
    return {"images": data["images"].tolist(), "classes": classes}
