"""Evaluation metrics for PathSnap map-matching.

Owned by: Sanjit (feat/sanjit)
Primary Metric: Observation-level Top-1 Accuracy
Secondary Metrics: Macro F1 (threshold 0.5), Log Loss, Route Continuity, Candidate Miss Rate
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def pred_arcs(df: pd.DataFrame, col: str = 'p') -> pd.DataFrame:
    """Select the candidate arc with the highest predicted score for each GPS observation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Candidate table containing at least ['tid', 'obs', 'arc', 'label', col].
    col : str, default='p'
        Column name containing model predicted probabilities.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing exactly one selected arc per (tid, obs).
    """
    if df.empty:
        return pd.DataFrame(columns=['tid', 'obs', 'arc', 'label', col])

    # Ensure stable reset index so idxmax matches positional rows accurately
    work = df.reset_index(drop=True)
    idx = work.groupby(['tid', 'obs'], sort=False)[col].idxmax()
    return work.loc[idx, ['tid', 'obs', 'arc', 'label']].sort_values(['tid', 'obs']).reset_index(drop=True)


def top1(df: pd.DataFrame, col: str = 'p') -> float:
    """Compute observation-level Top-1 Accuracy.
    
    Definition: Share of observations whose highest-scoring candidate is the reference arc (label == 1).
    This is the primary selection metric for PathSnap.
    """
    if df.empty:
        return 0.0
    preds = pred_arcs(df, col)
    if preds.empty or 'label' not in preds:
        return 0.0
    return float((preds['label'] == 1).mean())


def continuity(sel: pd.DataFrame, arcs_by_tid: Dict[str, np.ndarray]) -> float:
    """Compute route continuity.
    
    Definition: Share of consecutive distinct predicted arcs where 
    the end node of the previous arc equals the start node of the next arc:
        to_node(prev) == from_node(next)

    Parameters
    ----------
    sel : pd.DataFrame
        Pointwise predicted arcs table with columns ['tid', 'obs', 'arc'].
    arcs_by_tid : dict
        Mapping of tid -> arcs array (shape M x 2, [from_node, to_node]).

    Returns
    -------
    float
        Route continuity in [0.0, 1.0].
    """
    if sel.empty:
        return 0.0

    ok = 0
    tot = 0
    for tid, g in sel.sort_values('obs').groupby('tid'):
        if tid not in arcs_by_tid:
            continue
        a = g['arc'].to_numpy()
        if len(a) == 0:
            continue
        
        # Keep only distinct consecutive arcs (transitions)
        mask = np.r_[True, a[1:] != a[:-1]]
        a_distinct = a[mask]
        
        if len(a_distinct) <= 1:
            # Entire trip stayed on 1 arc or 1 point -> zero broken transitions
            continue
            
        A = arcs_by_tid[tid]
        # Transition validity: to_node(a[i]) == from_node(a[i+1])
        valid_transitions = (A[a_distinct[:-1], 1] == A[a_distinct[1:], 0]).sum()
        ok += int(valid_transitions)
        tot += (len(a_distinct) - 1)

    if tot == 0:
        return 1.0 if len(sel) > 0 else 0.0

    return float(ok / tot)


def candidate_miss_rate(df: pd.DataFrame) -> float:
    """Compute candidate miss rate.
    
    Definition: Share of observations whose reference arc is absent from the generated candidate set.
    """
    if df.empty or 'label' not in df:
        return 0.0
    has_pos = df.groupby(['tid', 'obs'])['label'].max()
    return float((has_pos < 1).mean())


def macro_f1_score(y_true: Union[np.ndarray, pd.Series], y_prob: Union[np.ndarray, pd.Series], threshold: float = 0.5) -> float:
    """Compute Macro F1 score on candidate rows at a specified decision threshold.
    
    Uses zero_division=0 to prevent UndefinedMetricWarning when dealing with imbalanced candidate sets.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))
