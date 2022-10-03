from typing import Dict
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

NUM_SDGS = 16


def multilabel_metrics(query_df: pd.DataFrame, val_df: pd.DataFrame, num_labels: int = NUM_SDGS,
                   sdg_id_col_name: str = "sdg_id",
                   sdg_pred_col_name: str = "sdg_pred",) -> Dict[str, int]:
    """
        Computes precision, recall, and F1 scores per SDGs, given two DataFrames:
        one with query/Ml output, another one with "golden" annotations.
        :param query_df: a dataframe with an `sdg_id` column and `eid` as an index
        :param val_df: a dataframe with an `sdg_id` column and `eid` as an index
        :param num_labels: number of SDGs under consideration
        :param sdg_id_col_name: column name for the SDG Id
        :return: a DataFrame with metrics
        """

    # take only those IDs with prediction that are present in the validation set
    query_map = query_df[query_df.index.isin(val_df.index)]

    # group predictions by ID and collect a list of predicted Goals per ID
    pred_grouped = pd.DataFrame(query_map.groupby(query_map.index)[sdg_id_col_name].apply(list)) \
        .rename(columns={sdg_id_col_name: sdg_pred_col_name})

    # group labels by ID and collect a list of labels per ID
    labels_qrouped = pd.DataFrame(val_df.groupby(val_df.index)[sdg_id_col_name].apply(list))

    # collect both in a same DataFrame
    pred_df = pred_grouped.join(labels_qrouped, how='inner')

    # binarize labels and compute multi-label metrics
    mlb = MultiLabelBinarizer(classes=range(num_labels + 1))
    binarized_targets = mlb.fit_transform(pred_df[sdg_id_col_name])
    binarized_preds = mlb.transform(pred_df[sdg_pred_col_name])

    res_df = {
        'precision_micro': precision_score(y_true=binarized_targets,
                                           y_pred=binarized_preds, average='micro'),
        'precision_macro': precision_score(y_true=binarized_targets,
                                           y_pred=binarized_preds, average='macro'),
        'recall_micro': recall_score(y_true=binarized_targets,
                                     y_pred=binarized_preds, average='micro'),
        'recall_macro': recall_score(y_true=binarized_targets,
                                     y_pred=binarized_preds, average='macro'),
        'f1_micro': f1_score(y_true=binarized_targets,
                             y_pred=binarized_preds, average='micro'),
        'f1_macro': f1_score(y_true=binarized_targets,
                             y_pred=binarized_preds, average='macro'),

    }

    return res_df