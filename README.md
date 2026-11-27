# Twitter Hashtag Prediction

This repository contains a TensorFlow sequence-to-sequence model that predicts
weather-related labels from a tweet. The labels describe the tweet's sentiment,
when the weather is occurring, and the kinds of weather mentioned.

## What the project currently does

[`twitter_hashtag.py`](./twitter_hashtag.py) is a single, end-to-end training and
demonstration script. Running it:

1. Reads `train.csv` and `contractions.pkl`.
2. Cleans the tweet text and converts each labeled row into a target sequence.
3. Splits the labeled rows into a 90% training set and a 10% validation set.
4. Fits text tokenizers on the training data and trains an encoder-decoder model
   for 10 epochs.
5. Saves the trained model to `s2s` and a diagram to `model_plot.png`.
6. Reloads the saved model, predicts labels for a small selection of validation
   tweets, and prints predicted and actual labels.

Despite the presence of `test.csv` and `sampleSubmission.csv`, the script does
**not** currently make predictions for `test.csv` or write a submission file.
Its printed examples come from the held-out portion of `train.csv`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `twitter_hashtag.py` | Data preparation, model definition, training, saving, and validation examples. |
| `train.csv` | Labeled training data: 77,946 rows with tweet text, metadata, and 24 label columns. |
| `test.csv` | Unlabeled competition/test data: 42,157 rows. Not used by the current script. |
| `sampleSubmission.csv` | Submission-format example with 24 output-label columns. Not written by the current script. |
| `contractions.pkl` | Pickled contraction-expansion mapping used during tweet cleaning. |
| `variableNames.txt` | Human-readable descriptions of the sentiment (`s`), timing (`w`), and weather-kind (`k`) label codes. |
| `s2s/` | Existing TensorFlow SavedModel files. The script trains and saves a model here before loading it. |

Keep the CSV files and `contractions.pkl` in the same working directory as the
script. The code opens these paths relative to the current working directory.

## Data and label format

The training CSV has four input/metadata columns (`id`, `tweet`, `state`, and
`location`) followed by 24 label columns:

- `s1`–`s5`: sentiment (one category).
- `w1`–`w4`: when the weather occurs (one category).
- `k1`–`k15`: weather kinds (potentially multiple categories).

The model uses only the `tweet` column as input; it does not use state or
location. For each training row, the script chooses the largest sentiment
column, the largest timing column, and every weather-kind column whose value is
at least `0.5`. It represents the result as a decoder sequence starting with
`sos` and ending with `eos`. The code's label-code-to-display-name mapping is
defined in `tar_lab_val` in `twitter_hashtag.py`; `variableNames.txt` documents
the label codes.

Tweet cleaning lowercases the text, removes colons, splits on spaces, retains
only alphabetic tokens, and expands tokens found in `contractions.pkl`.

## Model overview

The model is an encoder-decoder recurrent neural network implemented with
TensorFlow/Keras:

- An embedding layer converts tweet tokens into 500-dimensional vectors.
- Three stacked LSTM layers encode the tweet.
- A decoder LSTM generates the output-label sequence from the encoder state.
- An attention layer combines decoder outputs with encoder outputs.
- A softmax layer predicts the next target token.

Training uses RMSProp, sparse categorical cross-entropy, batch size 500, and 10
epochs. The script sets the sequence lengths to the most frequent input and
target lengths in the data, then pads sequences to those lengths.

## Environment setup (Windows PowerShell)

Use Python **3.11** for the pinned TensorFlow/Keras 2 dependency. These are
Python package dependencies, so `requirements.txt` is the installation
manifest; a `.env` file of application environment variables is not needed.

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python twitter_hashtag.py
```

If PowerShell activation is restricted, run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\twitter_hashtag.py
```

`plot_model` also requires the **Graphviz application** to be installed and its
`dot` executable available on `PATH`. The `pydot` Python package is included in
`requirements.txt`, but it does not install the Graphviz executable. The script
downloads the NLTK `wordnet` and `punkt` data when it starts, so its first run
needs network access.

The first run trains the model, which may take a long time and use substantial
memory depending on the machine. Keep the script's working directory set to the
repository root so its relative file paths resolve.

## Dependency choices

`requirements.txt` pins TensorFlow to 2.15.1 because the script uses the Keras 2
API (including loading a SavedModel directory and `keras.backend`). TensorFlow
installs its compatible Keras dependency. NumPy is constrained below 2 to stay
compatible with this TensorFlow release. The other packages are used for CSV
handling, train/validation splitting, tokenization, and model visualization.

## Current limitations and reproducibility notes

- The script always trains a new model before its inference demonstration; it
  is not an inference-only entry point.
- `test.csv` and `sampleSubmission.csv` are currently unused.
- The tokenizer word-index mappings and calculated sequence lengths are not
  saved as separate artifacts. The `s2s` model directory alone is therefore
  insufficient for a reliable standalone prediction workflow.
- Inference code locates specific layers by numeric index, so changes to the
  model architecture can require corresponding updates to that code.
- The final demo assumes the validation split contains at least 19 examples.
- `s2s` is the training output path. Running the script can replace or modify
  the model files already in that directory.

To turn this into a reusable predictor or generate a competition submission,
the tokenizer and preprocessing metadata should also be persisted, and an
explicit inference path should read `test.csv` and write the required
`sampleSubmission.csv`-compatible output.
