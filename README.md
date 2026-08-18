# Algorithm Exploration Dashboard

This project rebuilds the ranking-exploration dashboard from one required input:

`input/source_of_truth_merged.xlsx`

The repository includes a fully built dashboard in `output/`, so you can open it immediately after cloning. You only need to rebuild it after changing the input workbook or the code.

## Run from scratch

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_dashboard.py
```

The included and regenerated dashboard is:

`output/interactive_rank_comparison.html`

Open that HTML file in a browser. The page is standalone; it does not need a web server.

## Regenerate after changing the input

Replace `input/source_of_truth_merged.xlsx` with the updated workbook, then run:

```bash
python build_dashboard.py
```

The scripts regenerate the intermediate data under `data/` and the final HTML under `output/`.

The main implementation is in `code/`. The build entry point is `build_dashboard.py`.
