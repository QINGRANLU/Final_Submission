# UCL CDT x ASOS: Learning Graph Representations to Predict Customer Returns in Fashion Retail

If you have any problems using this codebase, please post in the
[General Slack channel](https://asos-gnn.slack.com/archives/C033191VCM6) for the project.

## Setup

### Google Colab

Google Colab does not appear to support the use of virtual environments (especially intuitively), so instead the file
`asos-gnn-returns-requirements-colab.txt` is provided to ensure we're collaborators are working with the same versions
of our core dependencies.

To use this file, place it in the root of your Google Drive, and in the early cells of the Colab notebook you're
working in, run the following:

```python
# Mount Google Drive to Colab runtime
from google.colab import drive
drive.mount('/content/gdrive')
```

Followed by

```bash
# Install project requirements to Colab runtime
! pip3 install -r /content/gdrive/MyDrive/cdt-gnn-returns/asos-gnn-returns-requirements-colab.txt
```

### MacOS

To as closely as possible replicate environments between collaborators, on MacOS we recommend using a
[Conda](https://www.anaconda.com/products/distribution) virtual environment.

Once Conda is installed, run in your shell:

```bash
conda init zsh
```

And do as the CLI says.

After this, you should be able to create a Conda environment for the project using the file
`./asos-gnn-returns-env-macos.yml` as follows:

```bash
conda env create -f asos-gnn-returns-env-macos.yml
```

After which, specific versions of Python, Pip, and all other dependencies should be installed.

## PyG experimental features

Depending on what features you need for your experiments, you might need to install the version of PyTorch Geometric documented as "latest". For this you can use the environment `asos-gnn-returns-env-macos-pynightly.yml`.

It should be possible to set both environments up and switch between them depending on whether you want to use a stable version of PyG or take advantage of the latest features, e.g.

```bash
conda activate asos-gnn-returns
conda deactivate
conda activate asos-gnn-returns-pygnightly
```