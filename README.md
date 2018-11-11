# EEG_ML
Some basic functionality for reading, plotting, and processing EEG data and
performing classification with a couple of different methods. Relies on

The read functionality here is designed to work with the data available at
https://archive.ics.uci.edu/ml/datasets/eeg+database

I only trained on the 'large' dataset (the pre-split train/test tar files).
To start, specify the base directory in which the extracted datasets are located,
I used a folder name 'large_data_set' in which you can find extracted SMNI_CMI_TRAIN
and SMNI_CMI_TEST folders. Using this setup allows some of the pre-defined
metadata fetching to work correctly, e.g., `sample_file_list()`.

First make sure to run `python setup.py develop` in the from the `eeg_ML` folder.
Then try running the `explore-EEG_data.ipynb` Jupyter notebook. The other notebooks
provide some further exploration of features as well as running the
classification approaches that I applied.
