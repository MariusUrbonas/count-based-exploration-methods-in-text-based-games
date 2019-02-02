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


## Evaluating our models 

To evaluate our models we ll use pyfiction used in Baselines for Reinforcement Learning in Text Games (https://ieeexplore.ieee.org/abstract/document/8576056). 

Make yourself a new enviroment:

```bash
conda create --name baselines python=3.7
conda activate baselines
```

Install pyfiction:

```bash
pip install pyfiction
```



