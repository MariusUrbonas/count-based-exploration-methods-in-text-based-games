# long-short-term-memes
Framework and model code for the paper "[Name TBD]", which was also used as a submission for [Microsoft TextWorld competition]( https://www.microsoft.com/en-us/research/project/textworld/).

# Installation

## TextWorld

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
pip install https://github.com/Microsoft/TextWorld/archive/master.zip
```

To create a game:

```bash
tw-make custom --world-size 5 --nb-objects 10 --quest-length 5 --seed 1234 --output tw_games/custom_game.ulx
```

To play the game:

```bash
tw-play tw_games/custom_game.ulx
```

## Training data

Get the data from the [CodaLab Competition](https://competitions.codalab.org/competitions/20865#participate-get_starting_kit) at: Participate tab > Files > Public Data.

Unzip it into the repo's root directory (where this file is) and name the folder `train` so that it gets ignored by git.

## LSTM-DQN

With the `textworld` conda environment activated, and making sure that `which pip` points to the pip inside conda:

```bash
pip install spacy torch
```

## Evaluating our models 

To evaluate our models we ll use pyfiction used in Baselines for Reinforcement Learning in Text Games (https://ieeexplore.ieee.org/abstract/document/8576056). 

Make yourself a new enviroment:

```bash
conda create --name baselines python=3.7
conda activate baselines
```

```bash
python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
```

Install pyfiction:

```bash
pip install pyfiction
```
Play game by:

```bash
python baselines/baselines_for_pyfiction/pyfiction_example.py
```


