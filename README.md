## Instructions for environment setup

1. Download conda for virtual environment management. See instructions here: https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions

2. Open a terminal and type `bash`. Run the following command to create a virtual environment:
    ```bash
    conda create -n "isye_6644" python=3.13.7 ipython
    ```

3. Activate the virtual environment:
    ```bash
    conda activate isye_6644
    ```

4. Install poetry (used for package management):
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

5. Export homepath:
    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

6. Verify poetry installation:
    ```bash
    poetry --version
    ```

7. Ensure you're in the main directory (where `pyproject.toml` is) and run:
    ```bash
    poetry install
    ```

8. Install ipykernel:
    ```bash
    pip install ipykernel
    ```

9. Register kernel for use in Jupyter:
    ```bash
    python -m ipykernel install --user --name=isye_6644
    ```

## Running the Analysis

Once setup is complete, open `analysis.ipynb` and select `isye_6644` as the kernel. Hit "Run All" to run simulations and generate all output used in the report. Tables are copied to clipboard and pasted into attached Excel spreadsheet, while figures are output in folder `figures`.

> **Note:** It will take about an hour to run the greedy simulations from scratch. You can skip generating the simulation results and uncomment
>           the second cell in section 4 to read saved output from the `data` folder.

## Results

Results are discussed in yahtzee_report.pdf.
