# O-Ring Measurement Script

## Usage
```
python3 diameter.py -h
usage: diameter.py [-h] [-t THRESHOLD] [-r RULER_THRESHOLD] -d DIRECTORY -o OUTPUT_DIRECTORY [-i] [-l RULER_LINE_HEIGHT] [-b] [-m]

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
  -l RULER_LINE_HEIGHT, --ruler-line-height RULER_LINE_HEIGHT
                        Limit on the height of a ruler line
  -b, --debug           Outputs debugging images (dark, light) in rundir
  -m, --marked-ruler    Measures distance between two pieces of red tape on the ruler
```

## Suggested Parameters

1. Black o-rings: `python3 diameter.py -d inputs -o results`
2. Clear o-rings: `python3 diameter.py -d inputs -o results -i`
