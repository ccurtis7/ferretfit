## ferretfit
[![Build Status](https://travis-ci.org/ccurtis7/ferretfit.svg?branch=master)](https://travis-ci.org/uwescience/ferretfit)

Ferretfit is a package under development for the analysis of ferret videos.

Ferretfit was originally developed with an observed flaw in an existing software (Noldus CatwalkXT) used in ferret catwalk experiments. While the program could extract certain features from tracking videos, it failed to capture the lateral motion ("wonkiness") present in some sick animals that could be used as a tool to distinguish between experimental groups. Because the Noldus program didn't provide users with access to the raw data, it made such calculations difficult.

### Current workflow

Currently, the analysis is performed in a two-step process: (1) ImageJ extraction of footprint coordinates from screenshots from the initial program and (2) feature calculation using a Jupyter notebook.

### ImageJ image processing

Users take a screenshot of the final output from the Noldus CatwalkXT analysis. Users must also make sure to crop the images to the area of interest e.g.:



The ImageJ macro converts the image to a binary image, and calculates coordinates of the footprints based on the boxes the initial software uses to label the footprints. The current weakness of this method is threefold: (1) overlapping footprints are sometimes not distinguishable if the Watershed algorithm doesn't catch it, (2) if any of the text overlaps, these can sometimes be caught as extra tracks, and (3) any small footprints sometimes are edited out in one of the erosion steps.

The macro outputs files as csvs that can be fed into the iPython script.

### iPython script

The provided example notebook shows how ferretfit can be used to provide a few additional parameters to measure wonkiness. The following parameters are currently calculated:

* Deviation: the standard deviation of y coordinates of footprints.
* Range: the entire range (max - min) of y coordinates of footprints.
* % RSD: the relative standard deviation (deviation / range) of y coordinates of footprints.
* Amplitude: the amplitude of the fit sine curve to the coordinates of footprints.
* Period: the period of the fit sine curve to the coordinates of footprints.
* Paw count: total number of footprints
* Paw density: number of footprints per 100 pixels in x direction.
* Cross count: total number of times the "wobbles" (the moving average curve cross the y average value)
* Cross density: number of "wobbles" per 100 pixels in x direction.
* Stride length: average x distance between footprints.
* Stride deviation: standard deviation x distances between footprints.
