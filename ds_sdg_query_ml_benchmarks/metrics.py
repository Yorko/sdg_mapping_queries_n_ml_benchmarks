import pandas as pd


def calc_metrics_per_sdg(
    query_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_sdgs: int = 16,
    sdg_id_col_name: str = "sdg_id",
) -> pd.DataFrame:
    """

    Computes precision, recall, and F1 scores per SDGs, given two DataFrames:
    one with query/Ml output, another one with "golden" annotations.

    :param query_df: a dataframe with an `sdg_id` column and `eid` as an index
    :param val_df: a dataframe with an `sdg_id` column and `eid` as an index
    :param num_sdgs: number of SDGs under consideration
    :param sdg_id_col_name: column name for the SDG Id

    :return: a DataFrame with metrics
    """

    prec_scores, recall_scores, f1_scores, supports = [], [], [], []

    # we consider only those EIDs that are present in the current
    # validation set
    query_map = query_df[query_df.index.isin(val_df.index)]

    for sdg_id in range(1, num_sdgs + 1):

        # select IDs for the current sdg_id
        query_ids = query_map[query_map[sdg_id_col_name] == sdg_id].index
        val_ids = val_df.loc[val_df[sdg_id_col_name] == sdg_id].index
        overlap = set(query_ids).intersection(val_ids)

        # support – is the number of papers from the validation set
        # that are captured by the queries
        support = len(overlap)

        # precision – is the share of papers from the query set
        # that are classified into the same SDG as in the validation set
        precision = len(overlap) / len(query_ids) if len(query_ids) else 0

        # recall – is the share of papers from the validation set
        # that are classified into the same SDG by the queries
        recall = len(overlap) / len(val_ids) if len(val_ids) else 0

        # F1 score is the harmonic mean of precision and recall
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision and recall)
            else 0
        )

        # add metrics for the current SDG id
        prec_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)
        supports.append(support)

    # construct the final DataFrame
    res_df = pd.DataFrame(
        {
            "precision": prec_scores,
            "recall": recall_scores,
            "f1": f1_scores,
            "support": supports,
        },
        index=range(1, num_sdgs + 1),
    )
    return res_df


def micro_average(metric_df, metric_col_name="precision", weight_col_name="support"):
    """

    :param metric_df: output of the `calc_metrics_per_sdg` function
    :param: weight_col_name: column name for the number of observations
    :param metric_col_name: either "precision", "recall", or "f1"
    :return: a micro-averaged value of the metric
    """
    return (metric_df[metric_col_name] * metric_df[weight_col_name]).sum() / metric_df[
        weight_col_name
    ].sum()


def macro_average(metric_df, metric_col_name="precision"):
    """

    :param metric_df: output of the `calc_metrics_per_sdg` function
    :param metric_col_name: either "precision", "recall", or "f1"
    :return: a macro-averaged value of the metric
    """
    return metric_df[metric_col_name].mean()
