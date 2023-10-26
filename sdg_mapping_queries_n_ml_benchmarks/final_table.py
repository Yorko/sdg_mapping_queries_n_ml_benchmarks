import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from validate_query_output_vs_val_set import validate


def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments
    :return: argparse.Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Validate SDG queries/ML output with an evaluation dataset."
    )
    parser.add_argument(
        "--path_to_query_outputs",
        type=str,
        help="Path to a folder with CSV files mapping paper IDs (Scopus EIDs) to SDG ids "
        + "where the mapping is done with SDG queries or an ML model.",
    )
    parser.add_argument(
        "--path_to_val_sets",
        type=str,
        help="Path to a folder with CSV files mapping paper IDs (Scopus EIDs)"
        + " to SDG ids where the mapping is done manually.",
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

    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Precision, recall or F1",
    )

    parser.add_argument(
        "--averaging",
        type=str,
        default="micro",
        help="micro or macro",
    )

    parser.add_argument(
        "--path_to_save_result",
        type=str,
        default="result.xlsx",
        help="Where to save the final table",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    results = []

    for query_output_path in tqdm(
        sorted(Path(args.path_to_query_outputs).glob("*.zip"))
    ):
        cur_metrics = []
        for val_set_path in tqdm(sorted(Path(args.path_to_val_sets).glob("*.zip"))):
            print(f"{query_output_path.stem} vs {val_set_path.stem}")

            try:
                metric_dict = validate(
                    path_to_query_output=query_output_path,
                    path_to_val_set=val_set_path,
                    sdgs_to_consider=args.sdgs_to_consider,
                    index_col_name=args.index_col_name,
                )
                metric_val = metric_dict[f"{args.metric}_{args.averaging}"]

            except:
                metric_val = -1

            cur_metrics.append(metric_val)
        results.append(cur_metrics)

    res_df = pd.DataFrame(
        results,
        index=[
            el.stem for el in sorted(Path(args.path_to_query_outputs).glob("*.zip"))
        ],
        columns=[el.stem for el in sorted(Path(args.path_to_val_sets).glob("*.zip"))],
    ).sort_index()

    res_df.to_excel(args.path_to_save_result)


if __name__ == "__main__":
    main()
