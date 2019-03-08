# Sample Submission LSTM-DQN
--------------------------------------------------------------------------------
This is a sample submission with training components for the competition.

## Requirements
* Python 3.6
* [PyTorch 0.4][pytorch_install]
* [spaCy with English Model][spacy_install]
* See `requirements.txt` for detailed information.

## Training
* GPU vs CPU: Enable or disable cuda by using `use_cuda` in `config.yaml`.
* We recommend to train the agent on a machine with a GPU.
* Save and load pretrained model: See `checkpoint` in `config.yaml`.
* To train a model on all data, run `python train.py $TRAINING_DATA/`, or to train on part of the data, e.g., on games where recipe only contains 1 ingredient, run `python train.py $TRAINING_DATA/*recipe1*.ulx`
* Only the `*.ulx` files are supported for training the LSTM-DQN.

NB: `$TRAINING_DATA` points to the extracted contents of `train.zip` downloaded from the competition website.

## Submitting to CodaLab
* Note that this LSTM-DQN code requires the docker image mentioned in `Dockerimage` for the submission. Check out `Dockerfile` to see how that Docker image was built.
* To test the submission locally, from within this folder, run
`python $STARTING_KIT/test_submission.py ./ $SAMPLE_DATA/*.ulx ./results/`
* Make sure your submission zip contains:
  * the `saved_models` folder that includes the checkpoint for your best model;
  * the `*.py` files;
  * the `config.yaml` file;
  * the `vocab.txt` file;
  * the `Dockerimage` file.

NB: `$SAMPLE_DATA` points to the extracted contents of `sample_games.zip` found in the starting kit, and `$STARTING_KIT`points to the extracted contents of `starting_kit.zip`.

[pytorch_install]: https://pytorch.org/get-started/locally/#start-locally
[spacy_install]: https://spacy.io/usage/
