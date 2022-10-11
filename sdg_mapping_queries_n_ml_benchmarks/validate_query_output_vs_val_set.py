import argparse
from typing import Tuple
from pprint import pprint
import pandas as pd
from metrics import multilabel_metrics


def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments
    :return: argparse.Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Validate SDG queries/ML output with an evaluation dataset."
    )
    parser.add_argument(
        "--path_to_query_output",
        type=str,
        help="Path to a CSV file mapping paper IDs (Scopus EIDs) to SDG ids "
        + "where the mapping is done with SDG queries or an ML model.",
    )
    parser.add_argument(
        "--path_to_val_set",
        type=str,
        help="Path to a CSV file mapping paper IDs (Scopus EIDs) to SDG ids where the mapping is done manually.",
    )

    parser.add_argument(
        "--index_col_name",
        type=str,
        default="eid",
        help="Column name for the paper ID (Scopus EID)",
    )

    parser.add_argument(
        "--sdgs_to_consider",
        type=int,
        nargs="+",
        help="A subset of SDGs to consider (by default, 16 SDGs from 1 to 16)",
    )

    args = parser.parse_args()

    return args


def read_data(
    path_to_query_output: str, path_to_val_set: str, index_col_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads in the query/model mapping DataFrame and the validation DataFrame
    :param path_to_query_output: path to a CSV file mapping paper IDs (Scopus EIDs) to SDG ids
                                 where the mapping is done with SDG queries or an ML model.",
    :param path_to_val_set: path to a CSV file mapping paper IDs (Scopus EIDs) to SDG ids where
                            the mapping is done manually.
    :param index_col_name: column name for the paper ID (Scopus EID)
    :return: a tuple with two DataFrames
    """
    query_mapping_df = pd.read_csv(path_to_query_output, index_col=index_col_name)

    val_set_df = pd.read_csv(path_to_val_set, index_col=index_col_name)

    return query_mapping_df, val_set_df


def main() -> None:

    args = parse_args()
    query_mapping_df, val_set_df = read_data(
        path_to_query_output=args.path_to_query_output,
        path_to_val_set=args.path_to_val_set,
        index_col_name=args.index_col_name,
    )

    metric_df = multilabel_metrics(query_df=query_mapping_df, val_df=val_set_df,
                                   sdgs_to_consider=args.sdgs_to_consider)

    pprint(metric_df)


if __name__ == "__main__":
    main()
