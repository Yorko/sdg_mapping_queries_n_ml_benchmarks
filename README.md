# sdg\_mapping\_queries\_n\_ml\_benchmarks

This code reproduces benchmarking experiments presented in Table 4 of the paper "Identifying research supporting the United Nations Sustainable Development Goals".

<img src='img/table4_paper_experiments.png' width=70%>

## Instructions

1. **"Getting data"**: the data is shared via [ICSR Lab](https://www.elsevier.com/icsr/icsrlab). ICSR Lab is intended for scholarly research only and is a cloud-based computational platform which enables researchers to analyze large structured datasets, including aggregated data from Scopus author profiles, PlumX Metrics, SciVal Topics, and [Peer Review Workbench](https://www.elsevier.com/connect/new-dataset-offers-unique-insights-into-peer-review). Upon successful [application](https://www.elsevier.com/icsr/icsrlab/how-to-apply), download the data from ICSR Lab and put it in the `data` folder so that the file structure looks like this: <br><br>
<img src='img/sdg_data_tree_structure.png' width=40%>

1. **Managing dependencies:** The code is run with Python 3.10 and only needs Pandas >= 1.4.3. You can either install the dependency manually with `pip` or run `poetry install` to install the dependencies in a virtual environment managed by [Poetry](https://python-poetry.org/docs/basic-usage/) which is a modern tool for dependency management and packaging in Python.

1. **Reproducing the results**. To get metrics for a particular SDG mapping dataset (e.g. "Elsevier 2022 SDG mapping") and a particular evaluation dataset (e.g. "Elsevier multi-label SDG dataset") you can run the following command:

```bash
poetry run python sdg_mapping_queries_n_ml_benchmarks/validate_query_output_vs_val_set.py \
--path_to_query_output data/sdg_mapping_output/06_els_sm_sdg_2022_mapping.csv.zip \
--path_to_val_set data/sdg_eval_sets/04_els_multilabel_sdg_eval_dataset.csv.zip
``` 
This will print precision, recall, F1 by SDGs along with their micro- and macro-averaged values:

```
# scores by SDG are ommitted
Micro-average precision = 0.742
Macro-average precision = 0.631
Micro-average recall = 0.787
Macro-average recall = 0.754
Micro-average f1 = 0.753
Macro-average f1 = 0.671
```

The last 2 lines stand for the values presented in Table 4 of the paper: micro- and macro-averaged F1 scores. 