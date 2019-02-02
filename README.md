# long-short-term-memes
Framework and model code for the paper "[Name TBD]", which was also used as a submission for [Microsoft TextWorld competition]( https://www.microsoft.com/en-us/research/project/textworld/).

# Installation

Follow these steps to set up TextWorld. Adapted from the [TextWorld repo](https://github.com/microsoft/textworld).

Get required system libraries. For macOS:

```bash
brew install libffi curl git
```

Create and activate a Conda environment:

```bash
conda create --name textworld python=3.7
conda activate textworld
```

Make sure the environment is activated by checking that the terminal input says `(textworld)` before continuing. Install Python packages using pip:

```bash
pip install textworld
```

(Wait for Microsoft to fix [this issue](https://github.com/Microsoft/TextWorld/issues/121))