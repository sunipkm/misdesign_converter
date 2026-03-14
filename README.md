# `misdesign_converter`

This package is a supplemental library to [`misdesigner`](https://github.com/sunipkm/misdesigner) that can be used to build a command line tool to process raw images from an imaging spectrograph.

# Workflow

The workflow involving the converter is as follows:

### Generate Instrument Model

Use `misdesigner` with a reference image from the instrument to align the images from the instrument to the image plane of the model, and export a JSON file describing the instrument model. This step needs to be performed only once for each instrument, unless any modifications are made to the instrument that may alter the optical path.

The products from this step are:

- Instrument model JSON file
- An `ImageFormatter` callable: A function that
  - takes in an image, performs any necessary crop and resize,
  - converts it into a `DataArray` with the correct dimensions for `MisCurveRemover`.

An example of this step can be found in [`examples/hms1-eclipse/tools/hmsa_eclipse.py`](examples/hms1-eclipse/tools/hmsa_eclipse.py).

### Generate Secondary Straightening Files

To remove any residual curvature after the initial line straightening performed by `MisCurveRemover`, follow the example in [`examples/hms1-eclipse/tools`](examples/hms1-eclipse/tools) to create individual line profiles for specific spectral lines, which can be used as a reference to remove any residual curvature.

The products from this step are:

- A set of line profile files (netCDF DataArrays) for specific spectral lines,
- A `settings.json` file describing the format of the line profile file names, and any wavelength offset to be applied to the line profiles. An example is available [here](examples/hms1-eclipse/components/line_profiles/settings.json).

### Write supporting routines

#### `ImageLoader`

Write an `ImageLoader` callable: A function that takes in a file path and returns an image. This is used to load the raw images from the instrument, and return a tuple with the following members:

- The image data as a 2D numpy array,
- The timestamp of the image in seconds since the epoch (Unix timestamp),
- The exposure time of the image in seconds,
- Detector temperature in Celsius,
- Additional metadata as an optional dictionary of (key, (value, optional comment)) pairs.

####  `TimestampGenerator`
Write a `TimestampGenerator` callable: A function that takes in a path and returns a timestamp in seconds since the Unix epoch.

#### `FinalizeFlat`
Write a `FinalizeFlat` callable: A function that takes in a `window` string selecting the window being processed, and a `Dataset` containing a timeseries of straightnened spectral images, and returns a `Dataset`. Insert any relevant metadata into the `Dataset` in this function.

#### `DetectorNoiseLoader`
In case you want to perform dark-bias subtraction, provide a `DetectorNoiseLoader` callable: A function that takes in a file path and optionally returns a `DetectorNoise` object. The `DetectorNoise` object contains an average `dark` image, an average `bias` image (these have to be the same shape as the raw images returned by the `ImageLoader`), and a read noise value.

Optionally, provide a `FileListGenerator` callable: A function that takes in a directory and returns a list of files that are processed. This is used to generate a list of files to be processed when the input is a directory.

### Create the Converter
Create a `L1Converter` object using the `create()` method. This method auto-generates an `argprse.ArgumentParser` object with the relevant inputs, parses the command line arguments, and returns a `L1Converter` object. The `L1Converter` object has a `process()` method that processes the input files and produces the output L1 data. An example of this step can be found in [`examples/hms1-eclipse/eclipse_l1_converter.py`](examples/hms1-eclipse/eclipse_l1_converter.py).