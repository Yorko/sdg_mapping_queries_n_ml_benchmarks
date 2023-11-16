import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

NUM_SDGS = 16


def multilabel_metrics(
    query_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_labels: int = NUM_SDGS,
    sdg_id_col_name: str = "sdg_id",
    sdg_pred_col_name: str = "sdg_id",
    sdgs_to_consider: list[int] = (),
) -> dict[str, int]:
    """
    Computes precision, recall, and F1 scores per SDGs, given two DataFrames:
    one with query/Ml output, another one with "golden" annotations.
    :param query_df: a dataframe with an `sdg_id` column and `eid` as an index
    :param val_df: a dataframe with an `sdg_id` column and `eid` as an index
    :param num_labels: number of SDGs under consideration
    :param sdg_id_col_name: column name for the SDG Id
    :param sgds_to_consider: if the validation should be limited to a subset of SDGs
                            (e.g. for Bergen queries)
    :return: a DataFrame with metrics
    """

    query_df = query_df.copy()
    val_df = val_df.copy()

    if not sdgs_to_consider:
        sdgs_to_consider = range(num_labels + 1)

    val_df = val_df[val_df[sdg_id_col_name].isin(sdgs_to_consider)]
    query_df = query_df[query_df[sdg_pred_col_name].isin(sdgs_to_consider)]

    le = LabelEncoder().fit(
        val_df[sdg_id_col_name].unique().tolist()
        + query_df[sdg_pred_col_name].unique().tolist()
    )
    val_df[sdg_id_col_name] = le.transform(val_df[sdg_id_col_name])
    query_df[sdg_pred_col_name] = le.transform(query_df[sdg_pred_col_name])

    mlb = MultiLabelBinarizer(classes=range(len(sdgs_to_consider)))

    # take only those IDs with prediction that are present in the validation set
    query_map = query_df[query_df.index.isin(val_df.index)]

    # group predictions by ID and collect a list of predicted Goals per ID
    pred_grouped = pd.DataFrame(
        query_map.groupby(query_map.index)[sdg_pred_col_name].apply(list)
    )
    # group labels by ID and collect a list of labels per ID
    labels_qrouped = pd.DataFrame(
        val_df.groupby(val_df.index)[sdg_id_col_name].apply(list)
    )

    if sdg_id_col_name == sdg_pred_col_name:
        sdg_pred_col_name += "_pred"
        pred_grouped = pred_grouped.rename(columns={sdg_id_col_name: sdg_pred_col_name})

    # we are ignoring IDs that are not there in the valdiation set
    # but we don not ignore the IDs present only in the val set. Hence "right" join below
    pred_df = pred_grouped.join(labels_qrouped, how="right").fillna("").apply(list)

    # binarize labels and compute multi-label metrics
    binarized_targets = mlb.fit_transform(pred_df[sdg_id_col_name])
    binarized_preds = mlb.transform(pred_df[sdg_pred_col_name])

    res_dict = {
        "precision_micro": round(
            100
            * precision_score(
                y_true=binarized_targets, y_pred=binarized_preds, average="micro"
            )
        ),
        "precision_macro": round(
            100
            * precision_score(
                y_true=binarized_targets, y_pred=binarized_preds, average="macro"
            )
        ),
        "recall_micro": round(
            100
            * recall_score(
                y_true=binarized_targets, y_pred=binarized_preds, average="micro"
            )
        ),
        "recall_macro": round(
            100
            * recall_score(
                y_true=binarized_targets, y_pred=binarized_preds, average="macro"
            )
        ),
        "f1_micro": round(
            100
            * f1_score(
                y_true=binarized_targets, y_pred=binarized_preds, average="micro"
            )
        ),
        "f1_macro": round(
            100
            * f1_score(
                y_true=binarized_targets, y_pred=binarized_preds, average="macro"
            )
        ),
    }

    return res_dict
