import pandas as pd
import numpy as np


def compute_entity_f05(gt_set: set, pred_set: set) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.
    """
    if len(gt_set) == 0:
        if len(pred_set) == 0:
            return 1.0
        else:
            return 0.0

    if len(pred_set) == 0:
        return 0.0

    tp = len(gt_set.intersection(pred_set))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def evaluate_predictions(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """
    Macro-averaged evaluation over all S1 entities in gt_df.
    gt_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
    pred_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
    """
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        raw_m = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        m_set = set(i.strip() for i in raw_m.split(",") if i.strip())
        gt_map[s1] = m_set

    pred_map = {}
    for _, row in pred_df.iterrows():
        s1 = row["source1_entity_id"]
        raw_m = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        m_set = set(i.strip() for i in raw_m.split(",") if i.strip())
        pred_map[s1] = m_set

    f05_scores = []
    precisions = []
    recalls = []

    for s1, gt_set in gt_map.items():
        pred_set = pred_map.get(s1, set())
        
        if len(gt_set) == 0 and len(pred_set) == 0:
            f05 = 1.0
            p = 1.0
            r = 1.0
        elif len(gt_set) == 0:
            f05 = 0.0
            p = 0.0
            r = 1.0  # Or 0.0 for precision/recall reporting
        elif len(pred_set) == 0:
            f05 = 0.0
            p = 1.0
            r = 0.0
        else:
            tp = len(gt_set.intersection(pred_set))
            p = tp / len(pred_set)
            r = tp / len(gt_set)
            if tp == 0:
                f05 = 0.0
            else:
                f05 = (1.25 * p * r) / (0.25 * p + r)

        f05_scores.append(f05)
        precisions.append(p)
        recalls.append(r)

    mean_f05 = float(np.mean(f05_scores))
    mean_p = float(np.mean(precisions))
    mean_r = float(np.mean(recalls))

    return {
        "macro_f05": mean_f05,
        "macro_precision": mean_p,
        "macro_recall": mean_r,
        "num_entities": len(gt_map)
    }


if __name__ == "__main__":
    # Test example from problem statement
    # S1-00001: pred [S2-00047, S2-00193, S3-00812], GT [S2-00047, S3-00812] -> F0.5 = 0.714
    gt_test = pd.DataFrame([
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00047,S3-00812"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S3-00004"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": ""},
    ])
    pred_test = pd.DataFrame([
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00047,S2-00193,S3-00812"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S3-00004"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": ""},
    ])

    res = evaluate_predictions(gt_test, pred_test)
    print("Test evaluation result:", res)
    # Expected S1-00001 F0.5 = 0.7142857, S1-00002 F0.5 = 1.0, S1-00003 F0.5 = 1.0. Mean = (0.7142857+1+1)/3 = 0.90476
    print(f"S1-00001 F0.5: {compute_entity_f05({'S2-00047', 'S3-00812'}, {'S2-00047', 'S2-00193', 'S3-00812'}):.4f}")
