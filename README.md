# An Exploration of Count-Based Exploration Methods in Text-Based Games
Framework and model code for the paper "An Exploration of Count-Based Exploration Methods in Text-Based
Games", which was also used as a submission for [Microsoft TextWorld competition]( https://www.microsoft.com/en-us/research/project/textworld/).

Paper is added to the repository under the name report.pdf

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

Download the English language model:

```bash
python -m spacy download en
```

### Training

Make sure to edit `config.yaml`; use the following naming convention for experiments: `yyyy_mm_dd_name_experiment`, e.g. `2019_02_20_leon_initial_experiments`.

To train on all games:

```bash
python train.py ../../train
```
```bash
train.py ../../train/some-game.ulx -c  config.yaml
```

This should train on multiple games (needs to be tested). (Make a different folder with a subset of games to test?)
