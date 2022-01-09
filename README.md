# O-Ring Measurement Script

## Usage
```
python3 diameter.py -h
usage: diameter.py [-h] [-t THRESHOLD] [-r RULER_THRESHOLD] -d DIRECTORY -o OUTPUT_DIRECTORY [-i]

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        RGB threshold for o-ring color
  -r RULER_THRESHOLD, --ruler-threshold RULER_THRESHOLD
                        RGB threshold for rule color
  -d DIRECTORY, --directory DIRECTORY
                        Folder in which the o-ring images are found
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Folder to put processed images and csv in
  -i, --invert          Expect a ligher o-ring on a darker background
```

## Suggested Parameters

1. Black o-rings: `python3 diameter.py -d inputs -o results`
2. Clear o-rings: `python3 diameter.py -d inputs -o results -i`
