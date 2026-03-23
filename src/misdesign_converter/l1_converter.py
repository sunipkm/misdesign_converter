from __future__ import annotations
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timezone
import gc
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from argparse import Namespace, ArgumentParser
from alive_progress import alive_bar
from misdesigner import MisCurveRemover, MisInstrumentModel
import numpy as np
from numpy import ndarray
from xarray import DataArray, Dataset, Variable, concat as xr_concat
from uuid import uuid1
import psutil
from .secondary_straightening import SecondaryStraighten
from .flatfield import FlatFieldCorrector
from importlib.metadata import version

__version__ = version(str(__package__))

LOGGER = logging.getLogger(__name__)

PROCESS = psutil.Process()


def find_outlier_pixels(
    data: ndarray,
    tolerance: int | float = 3,
    edge_compensation: bool = True
) -> Tuple[ndarray, ndarray]:
    """This function finds the hot or dead pixels in a 2D dataset. Tolerance is the number of standard deviations used to cutoff the hot pixels. If you want to ignore the edges and greatly speed up the code, then set edge_compensation to False. The function returns a list of hot pixels and also an image with with hot pixels removed.

    Args:
        data (ndarray): 2D array of pixel values.
        tolerance (int | float, optional): Number of standard deviations used to cut off hot pixels. Defaults to 3.
        edge_compensation (bool, optional): Whether to compensate for edges. Defaults to True. Setting this to False will ignore the edges and greatly speed up the code, but may miss hot pixels on the edges.

    Returns:
        Tuple[ndarray, ndarray]: A tuple containing the image with hot pixels removed and the list of hot pixels.
    """

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = tolerance*np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    # because we ignored the first row and first column
    hot_pixels = np.array(hot_pixels) + 1

    # This is the image with the hot pixels removed
    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if edge_compensation == True:
        height, width = np.shape(data)

        ### Now get the pixels on the edges (but not the corners)###

        # left and right sides
        for index in range(1, height-1):
            # left side:
            med = np.median(data[index-1:index+2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # right side:
            med = np.median(data[index-1:index+2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width-1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width-1):
            # bottom:
            med = np.median(data[0:2, index-1:index+2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0, index] = med

            # top:
            med = np.median(data[-2:, index-1:index+2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height-1], [index]]))
                fixed_image[-1, index] = med

        ### Then the corners###

        # bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width-1]]))
            fixed_image[0, -1] = med

        # top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [0]]))
            fixed_image[-1, 0] = med

        # top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [width-1]]))
            fixed_image[-1, -1] = med

    return fixed_image, hot_pixels


TimestampGenerator = Callable[[Path], float]
"""
Generates a timestamp from a file path. The timestamp is in seconds since UNIX epoch 1970-01-01 00:00:00 UTC.
"""

FileListGenerator = Callable[[
    Path, TimestampGenerator, str], Tuple[List[Path], Optional[Any]]]
"""Generates a list of file paths with a given extension provided a directory and a timestamp generator. The file paths are sorted by the timestamp generated from the file paths. See get_filelist() for an implementation of this function.
"""


def handle_compression(file: Path) -> Tuple[Path, TemporaryDirectory]:
    if file.suffix == '.zip':
        mode = 'zip'
    elif file.suffix == '.tgz' or file.suffixes[-2:] == ['.tar', '.gz']:
        mode = 'tar'
    elif file.suffix == '.txz' or file.suffixes[-2:] == ['.tar', '.xz']:
        mode = 'tar'
    elif file.suffixes[-2:] == ['.tar', '.bz2']:
        mode = 'tar'
    else:
        raise ValueError(f'Unsupported file type: {file.suffix}')
    tempdir = TemporaryDirectory()
    if mode == 'zip':
        import zipfile
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(tempdir.name)
    elif 'tar' in mode:
        import tarfile
        # kind = mode.split('.')[-1]
        with tarfile.open(str(file), 'r') as tar_ref:
            tar_ref.extractall(tempdir.name)
    return Path(tempdir.name), tempdir


def get_filelist(
    dir: Path,
    timestamp: TimestampGenerator,
    ext: str = 'fit'
) -> Tuple[List[Path], Optional[Any]]:
    """Enumerate files with extension in subdirectories to provided directory.

    Args:
        dir (Path): Directory whose subdirectories are enumerated for files with extension `ext`.
        timestamp (TimestampGenerator): Function to extract a timestamp from a file path. The timestamp is nanoseconds elapsed since UNIX epoch 1970-01-01 00:00:00 UTC.
        ext (str, optional): File extension to enumerate. Defaults to 'fit'.

    Returns:
        Tuple[List[Path], Optional[Any]]: A tuple containing the list of file paths with extension `ext` in subdirectories of `dir`, sorted by creation time, and an optional additional value, and an optional context object that may need to live as long as the file list is being used. The additional value is not used by the default implementation, but it can be used by custom implementations of FileListGenerator to return additional information along with the file list.
    """
    context = None
    if not dir.exists():
        raise ValueError(f'Directory {dir} does not exist.')
    if not dir.is_dir():
        dir, context = handle_compression(dir)
    subdirs = [d for d in dir.iterdir() if d.is_dir()]
    filelist = []
    for subdir in subdirs:
        files = list(subdir.glob(f'*.{ext}'))
        filelist.extend(files)
    if len(filelist) == 0:
        raise ValueError(f'No files with extension "{ext}" found in {dir}.')
    filelist.sort(key=timestamp)
    LOGGER.info(
        f'Found {len(filelist)} files in {dir} with extension "{ext}".')
    start_date = datetime.fromtimestamp(
        timestamp(filelist[0]), tz=timezone.utc)
    end_date = datetime.fromtimestamp(
        timestamp(filelist[-1]), tz=timezone.utc)
    LOGGER.info(
        f'First image: {start_date:%Y-%m-%d %H:%M:%S}, last image: {end_date:%Y-%m-%d %H:%M:%S}.')
    return filelist, context


@dataclass
class DetectorNoise:
    """
    Data class to hold dark and bias values.
    """
    dark: ndarray
    bias: ndarray
    readnoise: float


DetectorNoiseLoader = Callable[[Path], Optional[DetectorNoise]]
"""
Loader function for detector noise. Takes a file path and an optional read noise value, and returns a DetectorNoise object.
"""


ImageLoader = Callable[
    [Path],
    Tuple[
        ndarray,
        float,
        float,
        float,
        Dict[
            str,
            Tuple[
                Any,
                Optional[str]
            ]
        ]
    ]
]
"""Loads an image from the given path and returns a tuple of (data, timestamp, exposure time, temperature, metadata). 
Timestamp is in seconds since UNIX epoch 1970-01-01 00:00:00 UTC. 
Exposure time is in seconds. 
Temperature is in Celsius.
Metadata is a dictionary of additional information, where each value is a tuple of (value, description).
The metadata is assumed to be time-dependent, and will be stored as a time series.
"""

ImageFormatter = Callable[[ndarray, MisCurveRemover], DataArray]
"""Formats the input image for use with the curve remover model. This may include cropping, resizing, or other preprocessing steps. The output should be a DataArray that can be directly fed into the curve remover model.
"""


@dataclass
class ImageFile:
    """Data class to hold image file information.
    """
    data: ndarray
    noise: Optional[ndarray]
    tstamp: float
    exposure: float
    temperature: float
    metadata: Dict[
        str,
        Tuple[Any, Optional[str]]
    ] = field(default_factory=dict)

    @staticmethod
    def load(
        file: Path,
        loader: ImageLoader,
        detector_noise: Optional[DetectorNoise],
        calculate_noise: bool,
    ) -> ImageFile:
        """Load a file using the provided read function and apply noise correction if detector noise is provided.

        Args:
            - file (Path): File path to load.
            - loader (ImageLoader): Function to load data from a file path. The function should return a tuple of (data, timestamp, exposure time, temperature, metadata). Timestamp is in seconds since UNIX epoch 1970-01-01 00:00:00 UTC. Exposure time is in seconds. Temperature is in Celsius. Metadata is a dictionary of additional information, where each value is a tuple of (value, description).
            - detector_noise (Optional[DetectorNoise]): Detector noise information.
            - calculate_noise (bool): Whether to calculate noise based on the detector noise information. If False, the noise will not be calculated and will be set to None.

        Returns:
            ImageFile: An ImageFile object containing the loaded data, noise (if calculated), timestamp, exposure time, temperature, and metadata.
        """
        data, tstamp, exposure, temperature, metadata = loader(file)

        if detector_noise is not None:
            if calculate_noise:
                rn = detector_noise.readnoise
                noise = data + rn * rn
                noise = np.sqrt(noise) / exposure
            else:
                noise = None
            data = data - detector_noise.bias - detector_noise.dark * exposure
        else:
            noise = None
        data /= exposure
        return ImageFile(
            data=data,
            noise=noise,
            tstamp=tstamp,
            exposure=exposure,
            temperature=temperature,
            metadata=metadata
        )

    @property
    def nbytes(self) -> int:
        """Calculate the size of the image data and noise in bytes.
        """
        return self.data.nbytes \
            + (self.noise.nbytes if self.noise is not None else 0) \
            + 8*3

    @property
    def timestamp(self) -> datetime:
        """Convert the timestamp to a datetime object.

        Returns:
            datetime: Timestamp of the image in UTC.
        """
        return datetime.fromtimestamp(self.tstamp, tz=timezone.utc)

    def process(
        self,
        windows: List[str] | Tuple[str],
        model: MisCurveRemover,
        ifilter: ImageFormatter,
    ) -> List[Dataset]:
        """Process an image.
        The image is cropped, resizeed, and straightened using the provided model. The output is a dictionary of straightened images for each window.

        Args:
            windows (List[str]): List of windows.
            model (MisCurveRemover): Curve remover model.
            ifilter (ImageFormatter): Function to prepare the input image for use with the curve remover model.

        Returns:
            List[Dataset]: Curve-removed output for each window
        """
        inp = ifilter(self.data, model)
        if self.noise is not None:
            noise = ifilter(self.noise, model)
        else:
            noise = None
        result = []
        for window in windows:
            data = model.straighten_image(inp, window, coord='Slit')
            image = data.expand_dims(
                dim={'tstamp': (self.tstamp,)}
            ).to_dataset(name='intensity')
            image['intensity'].attrs['unit'] = 'ADU s^{-1}'
            if noise is not None:
                snoise = model.straighten_image(noise, window, coord='Slit').expand_dims(
                    dim={'tstamp': (self.tstamp,)}
                )
                image['noise'] = snoise
                image['noise'].attrs['unit'] = 'ADU s^{-1}'
                image['noise'].attrs['description'] = 'Estimated noise in ADU per second, based on shot noise, read noise, and dark noise.'
                image['intensity'].attrs['description'] = 'Intensity of the image after exposure normalization after dark and bias subtraction.'
            else:
                image['intensity'].attrs['description'] = 'Intensity in ADU per second.'

            image['exposure'] = Variable(
                dims='tstamp', data=np.array([self.exposure]), attrs={'unit': 's', 'description': 'Exposure time in seconds.'}
            )
            image['ccdtemp'] = Variable(
                dims='tstamp', data=np.array([self.temperature]), attrs={'unit': 'C', 'description': 'CCD temperature in Celsius.'}
            )
            for key, (value, description) in self.metadata.items():
                image[key] = Variable(
                    dims='tstamp',
                    data=np.array([value]),
                )
                if description is not None:
                    image[key].attrs['description'] = description
            result.append(image)
        return result


def process_output_size(output: List[Dataset]) -> int:
    """Calculate the size of the output datasets in bytes.

    Args:
        output (List[Dataset]): Output datasets for each window.
    Returns:
        int: Size of the output datasets in bytes.
    """
    size = 0
    for dataset in output:
        size += dataset.nbytes
    return size


FinalizeFlat = Callable[[str, Dataset], Dataset]
"""Finalize the flattened output datasets.
This function is called for each window with the window name and the concatenated
time series of straightened spectra for that window. The function will return a
finalized dataset for that window with the correct attributes, any updates to coordinates etc.
"""


def flatten_output(
    keys: List[str] | Tuple[str],
    output: deque[List[Dataset]],
    finalize: FinalizeFlat,
) -> Dict[str, Dataset]:
    """Flatten the output of multiple images into a single dataset for each window.

    Args:
        keys (List[str]): List of window names.
        output (List[List[Dataset]]): List of output datasets for each image.
        finalize (FinalizeFlat): Function to finalize the flattened output.
    Returns:
        Dict[str, Dataset]: Flattened output dataset for each window.
    """
    flattened = {}
    while output:
        dl = output.popleft()
        for key, dataset in zip(keys, dl):
            if key not in flattened:
                flattened[key] = []
            flattened[key].append(dataset)
    attr_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    for key in flattened.keys():
        ds: Dataset = xr_concat(flattened[key], dim='tstamp')
        ds = finalize(key, ds)
        ds.attrs.update(dict(
            ROI=f'{str(key)}',
            creation_date=attr_time,
        ))
        ds['tstamp'].attrs.update(dict(
            unit='s',
            description='Timestamp of the image in seconds since UNIX epoch 1970-01-01 00:00:00 UTC.'
        ))
        ds['gamma'].attrs.update(dict(
            unit='deg',
            description='Incident angle in degrees'
        ))
        flattened[key] = ds
    return flattened


@dataclass
class L1Converter:
    """L1 converter settings.
    """
    imagefiles: List[Path]
    outdir: Path
    prefix: str
    predictor: MisCurveRemover
    loader: ImageLoader
    ifilter: ImageFormatter
    flatfinalize: FinalizeFlat
    _start: date
    enable_noise: bool = False
    secondary: Optional[Tuple[str, Path]] = None
    flatfield: Optional[FlatFieldCorrector] = None
    detnoise: Optional[DetectorNoise] = None
    window: Optional[List[str]] = None
    memlimit: int = 1024
    filecontext: Optional[Any] = None
    _mem_used: int = 0

    @staticmethod
    def create(
        description: Optional[str],
        invocation_dir: Optional[Path],
        timestamp: TimestampGenerator,
        loader: ImageLoader,
        ifilter: ImageFormatter,
        flatfinzalize: FinalizeFlat,
        detnoiseloader: DetectorNoiseLoader,
        filelister: FileListGenerator = get_filelist,
        extension: str = 'fit'
    ) -> L1Converter:
        """Create a L1Converter object by parsing command line arguments.

        Args:
            description (Optional[str]): Description of the program to be shown in the help message. If None, a default description will be used.
            invocation_dir (Optional[Path]): Directory from which the program is invoked. This is used to resolve relative paths for model description file, detector dark/bias file and secondary straightening files. If None, it defaults to the current working directory.
            timestamp (TimestampGenerator): Timestamp generator.
            loader (ImageLoader): Image loader.
            ifilter (ImageFormatter): Image formatter.
            flatfinalize (FinalizeFlat): Function to finalize the flattened output.
            filelister (FileListGenerator, optional): File list generator. Defaults to get_filelist.
            extension (str, optional): File extension to look for. Defaults to 'fit'.

        Returns:
            L1Converter: L1Converter object created from command line arguments.
        """
        parser = L1Converter.parser(description)
        args = parser.parse_args()
        return L1Converter.from_parsed(
            invocation_dir,
            args,
            timestamp,
            loader,
            ifilter,
            flatfinzalize,
            detnoiseloader,
            filelister,
            extension
        )

    @staticmethod
    def parser(description: Optional[str]) -> ArgumentParser:
        """Create an argument parser for L1Converter.

        Args:
            description (Optional[str]): Description of the program to be shown in the help message. If None, a default description will be used.

        Returns:
            ArgumentParser: Argument parser. Call parse_args() on the returned parser to get the arguments, and then use from_parsed() to create a L1Converter object.
        """
        if description is None:
            desc = 'Convert MisDesign-ed instrument L0 data to L1 data, with exposure normalization and Dark subtraction. It uses the MisInstrument model to extact ROI and performs line straightening. This program will not work without the instrument definition JSON file.'
        else:
            desc = description
        parser = ArgumentParser(
            add_help=False,
            description=desc,
        )
        required_args = parser.add_argument_group('Required arguments')
        optional_args = parser.add_argument_group('Optional arguments')
        other_args = parser.add_argument_group('Other Options')

        required_args.add_argument(
            'rootdir',
            metavar='rootdir',
            type=str,
            help='Root directory containing HiT&MIS data')
        required_args.add_argument(
            'dest',
            nargs='?',
            help='Root directory where L1 data will be stored')
        required_args.add_argument(
            'dest_prefix',
            nargs='?',
            default=sys.argv[0].split('.')[0].replace(' ', '_'),
            help='Prefix of final L1 data file name')
        optional_args.add_argument(
            '--window',
            required=False,
            type=str,
            default='',
            help='Comma-separated list of windows to process. Default is all windows.')
        optional_args.add_argument(
            '--memory-limit',
            required=False,
            type=int,
            default=1024,
            help='Memory limit in MB for the output datasets. If the output exceeds this limit, it will be flattened into a single dataset for each window. Default is 1024 MB.'
        )
        optional_args.add_argument(
            '--dark',
            required=False,
            type=str,
            help='Dark data file path. Default is pixis_dark_bias.xz'
        )
        required_args.add_argument(
            '--model',
            required=True,
            type=str,
            help='Path to the instrument model file'
        )
        optional_args.add_argument(
            '--noise',
            action='store_true',
            help='Whether to add noise to the data. Default is False.'
        )
        optional_args.add_argument(
            '--flatfield',
            required=False,
            type=str,
            help='Flat field image file path. If not provided, flat field correction will not be applied.'
        )
        optional_args.add_argument(
            '--secondary',
            nargs='+',
            type=str,
            required=False,
            help='Enable secondary straightening. Requires the kind (check line profiles), and optionally the path to the line profile directory. If only the kind is specified, a directory named $PWD/components/line_profiles will be used. The line profile directory must contain .nc files with line profiles, and a settings.json file describing the separator used to discern the kind and window of the line profiles, and the central wavelength setting for each window and kind. If CWL settings are not provided, they will default to "none", i.e. no shift applied to the line profile.'
        )
        optional_args.add_argument(
            '-h', '--help', action='help', help='Show this help message and exit')
        other_args.add_argument(
            '--version',
            action='version',
            version=f'%(prog)s {__version__}'
        )
        return parser

    @classmethod
    def from_parsed(
        cls,
        invocation_dir: Optional[Path],
        args: Namespace,
        timestamp: TimestampGenerator,
        loader: ImageLoader,
        ifilter: ImageFormatter,
        flatfinalize: FinalizeFlat,
        detnoiseloader: DetectorNoiseLoader,
        filelister: FileListGenerator = get_filelist,
        extension: str = 'fit',
    ) -> L1Converter:
        """Create an L1Converter from parsed arguments.

        Args:
            args (Namespace): Parsed arguments.
            timestamp (TimestampGenerator): Timestamp generator.
            loader (ImageLoader): Image loader.
            ifilter (ImageFormatter): Image formatter.
            flatfinalize (FinalizeFlat): Function to finalize the flattened output.
            detnoiseloader (DetectorNoiseLoader): Function to load detector noise.
            filelister (FileListGenerator, optional): File list generator. Defaults to get_filelist.
            extension (str, optional): File extension to look for. Defaults to 'fit'.
            invocation_dir (Optional[Path], optional): Directory from which the program is invoked. This is used to resolve relative paths for the secondary straightening line profile directory. If None, it defaults to the current working directory.

        Raises:
            ValueError: Invalid number of arguments for --secondary. Should be 1 or 2.

        Returns:
            L1Converter: L1Converter object created from parsed arguments.
        """
        idir = invocation_dir if invocation_dir is not None else Path.cwd()
        window = args.window.split(',') if args.window else None
        rootdir = Path(args.rootdir)
        files, context = filelister(rootdir, timestamp, extension)
        if rootdir.is_file():
            rootdir = rootdir.parent
        outdir = Path(args.dest) if args.dest else rootdir / 'L1_data'
        outdir.mkdir(parents=True, exist_ok=True)
        destprefix = str(Path(args.dest_prefix).stem)
        if not Path(args.model).exists():
            model = idir / 'components' / args.model
        else:
            model = Path(args.model)
        if args.dark and not Path(args.dark).exists():
            dark = idir / 'components' / args.dark
        else:
            dark = Path(args.dark) if args.dark else None
        LOGGER.info(f'Invocation directory: {idir}')
        LOGGER.info(f'Source root directory: {rootdir}')
        LOGGER.info(f'Output directory: {outdir}, Prefix: {destprefix}')
        LOGGER.info(f'Model file: {model}')
        LOGGER.info(f'Dark file: {dark}')
        model = MisInstrumentModel.load(str(model.resolve()))
        predictor = MisCurveRemover(model)
        if args.secondary is None:
            secondary = None
        elif len(args.secondary) == 1:
            secondary = (
                str(args.secondary[0]),
                idir / 'components' / 'line_profiles'
            )
        elif len(args.secondary) == 2:
            secondary = (str(args.secondary[0]), Path(args.secondary[1]))
        else:
            raise ValueError(
                'Invalid number of arguments for --secondary. Should be 1 or 2.')
        if args.flatfield is None:
            flatfield = None
        else:
            flatfield = FlatFieldCorrector(Path(args.flatfield))
        limit = args.memory_limit
        if limit <= 0:
            limit = 0
        if secondary is not None:
            limit = limit // 2
        date = datetime.fromtimestamp(
            timestamp(files[0]), tz=timezone.utc).date()
        return cls(
            imagefiles=files,
            filecontext=context,
            outdir=outdir,
            prefix=destprefix,
            predictor=predictor,
            loader=loader,
            ifilter=ifilter,
            flatfinalize=flatfinalize,
            _start=date,
            enable_noise=args.noise,
            secondary=secondary,
            window=window,
            memlimit=limit,
            flatfield=flatfield,
            detnoise=detnoiseloader(dark) if dark is not None else None
        )

    def check_write(self, date: datetime, size: int, index: int) -> Optional[date]:
        """Check if writing a dataset of size `size` bytes at time `date` would exceed the memory limit.

        Args:
            date (datetime): Timestamp of the dataset to be written.
            size (int): Size of the dataset to be written in bytes.
            index (int): Index of the current image being processed.
        Returns:
            Optional[date]: The date if writing the dataset would exceed the memory limit, None otherwise.
        """
        if index == len(self.imagefiles) - 1:
            # If this is the last image, we want to write regardless of memory usage to ensure all data is written.
            return date.date()
        # Memory limit check
        if self.memlimit > 0 and self._mem_used + size > self.memlimit * (1 << 20):
            self._mem_used = 0
            return self._start
        # Date check
        if date.date() != self._start:
            ddate = self._start
            self._start = date.date()
            self._mem_used = 0
            return ddate
        self._mem_used += size
        return None

    @property
    def memory(self) -> int:
        """Get the current memory usage in bytes.

        Returns:
            int: Current memory usage in bytes.
        """
        return self._mem_used

    def process(self):
        """Process the files.
        """
        if self.window is None:
            windows = list(self.predictor.windows)
        else:
            windows = self.window
        outputs = deque()
        filedict: Dict[str, Dict[str, List[Tuple[Path, str]]]] = {}
        tempmgr = TemporaryDirectory()
        tempdir = Path(tempmgr.name)
        mdigit = len(str(self.memlimit))
        secondary = None
        if self.flatfield is not None:
            windows = list(set(windows).intersection(
                set(self.flatfield.windows)))
        if self.secondary is not None:
            kind = self.secondary[0]
            manager = SecondaryStraighten(self.secondary[1])
            windows = list(set(windows).intersection(
                set(manager.windows(kind))))
            secondary = (
                kind,
                manager,
            )
        LOGGER.info(self.status(windows))
        try:
            with alive_bar(len(self.imagefiles)) as bar:
                for fidx, file in enumerate(self.imagefiles):
                    if self.memlimit > 0:
                        bar.text = f'{self._start:%Y-%m-%d} [{self.memory / (1 << 20):<{mdigit}.2f} MiB | {PROCESS.memory_info().rss / (1 << 20):<{mdigit}.2f} MiB]'
                    else:
                        bar.text = f'{self._start:%Y-%m-%d} [{self.memory / (1 << 20):.2f} MiB | {PROCESS.memory_info().rss / (1 << 20):.2f} MiB]'
                    # Load the image and pre-process it
                    image = ImageFile.load(
                        file, self.loader, self.detnoise, self.enable_noise)
                    # Process the image and get the output datasets for each window
                    output = image.process(
                        windows, self.predictor, self.ifilter)
                    # Add the output datasets to the list of outputs and check if we need to write the output datasets to disk
                    outputs.append(output)
                    # Calculate the size of the output datasets in bytes
                    output_size = process_output_size(output)
                    # Check if data needs to be written to disk based on memory limit and date change
                    wdate = self.check_write(
                        image.timestamp, output_size, fidx)
                    bar()
                    if wdate is not None:
                        bar.text = f'{self._start:%Y-%m-%d} [ _ MiB | {PROCESS.memory_info().rss / (1 << 20):.2f} MiB] Flattening...'
                        output = flatten_output(
                            windows, outputs, self.flatfinalize)
                        gc.collect()
                        if self.flatfield is not None:
                            bar.text = f'{self._start:%Y-%m-%d} [ _ MiB | {PROCESS.memory_info().rss / (1 << 20):.2f} MiB] Applying flat field correction...'
                            for key in output.keys():
                                # bar.text = f'Applying flat field correction to {key}...'
                                output[key] = self.flatfield.apply(
                                    output[key], key)
                                bar.text = f'{self._start:%Y-%m-%d} [ _ MiB | {PROCESS.memory_info().rss / (1 << 20):.2f} MiB] Finished flat field correction for {key}.'
                        if secondary:
                            bar.text = f'{self._start:%Y-%m-%d} [ _ MiB | {PROCESS.memory_info().rss / (1 << 20):.2f} MiB] Secondary straightening...'
                            for key in output.keys():
                                # bar.text = f'Applying secondary straightening to {key}...'
                                output[key] = secondary[1].straighten(
                                    output[key], secondary[0], key
                                )
                                bar.text = f'{self._start:%Y-%m-%d} [ _ MiB | {PROCESS.memory_info().rss / (1 << 20):.2f} MiB] Finished secondary straightening for {key}.'
                        gc.collect()
                        wdate = wdate.strftime('%Y%m%d')
                        for key, dataset in output.items():
                            encoding = {
                                var: {'zlib': True, 'complevel': 5}
                                for var in (
                                    *dataset.data_vars.keys(), *dataset.coords.keys()
                                )
                            }
                            temppath = tempdir / f'{uuid1().hex}.nc'
                            outfile = f'{self.prefix}_{key}_{wdate}'
                            bar.text = f'Writing {outfile}...'
                            dataset.to_netcdf(
                                temppath, encoding=encoding, engine='netcdf4'
                            )
                            bar.text = f'{self._start:%Y-%m-%d} [ _ MiB | {PROCESS.memory_info().rss / (1 << 20):.2f} MiB] Finished writing {outfile}.'
                            if wdate not in filedict:
                                filedict[wdate] = {}
                            if key not in filedict[wdate]:
                                filedict[wdate][key] = []
                            filedict[wdate][key].append((temppath, outfile))

            # Move the temporary files to the final destination
            LOGGER.info('Moving temporary files to final destination...')
            for wdate, keyfiles in filedict.items():
                for key, files in keyfiles.items():
                    ndigit = len(str(len(files)))
                    outdir = self.outdir / wdate
                    outdir.mkdir(parents=True, exist_ok=True)
                    for fidx, (temppath, outfile) in enumerate(files):
                        dest = outdir / f'{outfile}[{fidx:0{ndigit}d}].nc'
                        temppath.rename(dest)

            LOGGER.info('Processing completed.')
        except KeyboardInterrupt:
            LOGGER.warning(
                'Processing interrupted by user.'
            )

    def status(self, windows: List[str]) -> str:
        output = [f'Processing {len(self.imagefiles)} files']
        output += [f'Window: {", ".join(windows)}']
        if self.detnoise is not None:
            output += ['Detector noise correction enabled']
        else:
            output += ['Detector noise correction disabled']
        if self.enable_noise:
            output += ['Noise calculation enabled']
        else:
            output += ['Noise calculation disabled']
        if self.secondary is not None:
            output += [
                f'Secondary straightening enabled with kind "{self.secondary[0]}"']
        else:
            output += ['Secondary straightening disabled']
        if self.memlimit > 0:
            output += [f'Memory limit: {self.memlimit} MiB']
        return ', '.join(output)

    @staticmethod
    def count_nmove(filedict: Dict[str, Dict[str, List[Tuple[Path, str]]]]) -> int:
        """Count the total number of files and total size of files to be moved.

        Args:
            filedict (Dict[str, Dict[str, List[Tuple[Path, str]]]]): Dictionary of files to be moved.
        Returns:
            int: Total number of files to be moved.
        """
        nfiles = 0
        for _, keyfiles in filedict.items():
            for _, files in keyfiles.items():
                nfiles += len(files)
        return nfiles
