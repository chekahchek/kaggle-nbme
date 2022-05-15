mkdir input
cd input
kaggle competitions download -c nbme-score-clinical-patient-notes
unzip -qq nbme-score-clinical-patient-notes.zip -d nbme-score-clinical-patient-notes
rm nbme-score-clinical-patient-notes.zip
kaggle datasets download -d chekhui/electralarge
unzip -qq electralarge.zip -d electralarge
rm electralarge
kaggle datasets download -d chekhui/debertav3largeretrained
unzip -qq debertav3largeretrained.zip -d debertav3largeretrained
rm debertav3largeretrained
kaggle datasets download -d chekhui/debertalargepl
unzip -qq debertalargepl.zip -d debertalargepl
rm debertalargepl
kaggle datasets download -d chekhui/debertav2xlarge
unzip -qq debertav2xlarge.zip -d debertav2xlarge
rm debertav2xlarge
kaggle datasets download -d chekhui/debertav1large
unzip -qq debertav1large.zip -d debertav1large
rm debertav1large