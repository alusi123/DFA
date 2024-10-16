# Dual-stream Feature Augmentation for Domain Generalization [ACM MM 2024]

## Data Preparation
Download the PACS dataset from the https://github.com/BIT-DA/CIRL/blob/main/README.md or from https://drive.google.com/file/d/1HNO090OHt3dzxY0qm2ukoaSqW_jSoS6P/view?usp=drive_link and extract the PACS under `$dataset`. The directory structure should look like
```
dataset/
|-- PACS/
|   |–– kfold/
|   |   |–– art_painting
|   |   |–– cartoon
|   |   |–– photo
|   |   |–– sketch
|   |–– datalists/
|   |   |–– art_painting_train.txt
|   |   |–– art_painting_val.txt
|   |   |–– art_painting_test.txt
|   |   |–– cartoon_train.txt
|   |   |–– ...
```

## Test
```
python shell_test.py -t=art_painting
```

## Train
```
python shell_train.py -t=art_painting
```
