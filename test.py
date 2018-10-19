from mne.datasets import eegbci

subject = 1
runs = [1]  # motor imagery: hands vs feet

# The data will be downloaded (approximately 7.5 MB)
#raw_fnames = eegbci.load_data(subject, runs,"./data")

from mne.io import concatenate_raws, read_raw_edf

#raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw=read_raw_edf("./data/S001R06.edf",preload=True)
#raw = concatenate_raws(raw_files)

raw.plot()

print(raw.info)

from mne import find_events
from mne.viz import plot_events

event_id = dict(hands=2, feet=3)
events = find_events(raw, shortest_event=0, stim_channel='STI 014')
plot_events(events, raw.info['sfreq'], event_id=event_id);


from mne import Epochs, pick_types

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

tmin, tmax = -1., 4.
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True)

from sklearn.pipeline import Pipeline

clf = Pipeline([('CSP', csp), ('LDA', lda)])
print(clf)

# (train will be done only between 1 and 2s)

epochs_train = epochs.copy().crop(tmin=1., tmax=2.)

epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

from sklearn.model_selection import ShuffleSplit, cross_val_score

# Define a monte-carlo cross-validation generator (reduce variance):
labels = epochs.events[:, -1] - 2
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
print(scores)

csp.fit_transform(epochs_data, labels)

from mne.channels import read_layout
layout = read_layout('EEG1005')

import numpy as np

csp.fit_transform(epochs_data, labels)

evoked = epochs.average()
evoked.data = csp.patterns_.T
evoked.times = np.arange(evoked.data.shape[0])

evoked.plot_topomap(times=[0, 1, 2, 3, 4, 5], ch_type='eeg', layout=layout,
                    scale_time=1, time_format='%i', scale=1,
                    unit='Patterns (AU)', size=1.5);





