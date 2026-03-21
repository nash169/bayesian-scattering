# Bayesian Scattering
A Bayesian scattering baseline for uncertainty quantification.

<table>
  <tr>
    <td valign="top" align="center">
      <strong>Histology</strong>
      <table>
        <tr>
          <td><img src="media/histology_train_1.png" alt="Histology training sample 1" width="110"></td>
          <td><img src="media/histology_train_2.png" alt="Histology training sample 2" width="110"></td>
        </tr>
        <tr>
          <td><img src="media/histology_test_1.png" alt="Histology test sample 1" width="110"></td>
          <td><img src="media/histology_test_2.png" alt="Histology test sample 2" width="110"></td>
        </tr>
      </table>
    </td>
    <td valign="top" align="center">
      <strong>Skin Lesion</strong>
      <table>
        <tr>
          <td><img src="media/skin_train_1.png" alt="Skin lesion training sample 1" width="110"></td>
          <td><img src="media/skin_train_2.png" alt="Skin lesion training sample 2" width="110"></td>
        </tr>
        <tr>
          <td><img src="media/skin_test_1.png" alt="Skin lesion test sample 1" width="110"></td>
          <td><img src="media/skin_test_2.png" alt="Skin lesion test sample 2" width="110"></td>
        </tr>
      </table>
    </td>
    <td valign="top" align="center">
      <strong>Asset</strong>
      <table>
        <tr>
          <td><img src="media/asset_train_1.png" alt="Asset training sample 1" width="110"></td>
          <td><img src="media/asset_train_2.png" alt="Asset training sample 2" width="110"></td>
        </tr>
        <tr>
          <td><img src="media/asset_test_1.png" alt="Asset test sample 1" width="110"></td>
          <td><img src="media/asset_test_2.png" alt="Asset test sample 2" width="110"></td>
        </tr>
      </table>
    </td>
  </tr>
</table>

## Getting Started
Install the package with `pip`:
```sh
pip install -e .
```

There is also a `uv` option if you prefer to manage the environment that way:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

Without installing the package, you can run the examples and benchmarks as Python modules:
```sh
python -m <path/to>.<script>
```
If you prefer to run the examples or benchmarks in JupyterLab, generate the `.ipynb` file:
```sh
jupytext --to ipynb <script>.py
```
Optionally, you can pair and sync the `.ipynb` and `.py` files to keep both updated:
```sh
jupytext --set-formats ipynb,py:percent <script>.ipynb
```
To run either the examples or the benchmarks, you need to specify three paths in each script:
```python
os.environ["DATA_PATH"] =
os.environ["FEATURES_PATH"] =
os.environ["RESULTS_PATH"] =
```
The first defines where the datasets are stored. The second defines where the generated features are stored, which avoids recomputing them every time you run a script. The third defines where the results are saved.
