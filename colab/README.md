# ElectroPatch in Google Colab

Open [ElectroPatch_Comparison.ipynb](ElectroPatch_Comparison.ipynb) in Colab after this repository is public. The notebook clones this repository, installs its Python package with the ML extra, runs the same `electropatch compare` and `electropatch ml` commands as the local workflow, and displays the eight-structure tables and plots.

1. Use the **Open in Colab** badge near the top of the repository README. The repository owner must first replace `YOUR_USERNAME` in that badge URL.
2. In Colab, select **Runtime → Run all**.
3. When prompted, paste the public repository URL, for example `https://github.com/YOUR_USERNAME/electropatch`.
4. Inspect the fixed-score and grouped-ML comparison figures and per-structure plots. Change `PDB_ID` in the detail cell to any ID shown in the table.
5. Choose `STORAGE_MODE` in the notebook's first code cell before running:

| Mode | Result |
| --- | --- |
| `"temporary"` (default) | Keeps CSV, JSON and figures in the Colab runtime and creates a downloadable ZIP in its Files panel. These disappear when the runtime is deleted. |
| `"drive"` | Asks to mount Google Drive and saves each run under `MyDrive/ElectroPatch/runs/`. Files persist after the runtime ends. |
| `"off"` | Uses temporary files for calculation and display, then deletes them in the final cell. No Drive copy or ZIP is made. |


The notebook contains no independent scoring implementation. It always calls the package in this repository and uses its frozen `data/manifest.csv` and PDB files, keeping Colab and local calculations aligned. No GPU is required. You can also open it from a local Jupyter session launched in the repository folder.

For interpretation and exact local reproduction steps, read [the analysis](../docs/ANALYSIS.md) and [reproducibility record](../docs/REPRODUCIBILITY.md).
