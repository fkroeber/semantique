# tbd: see vrt.py (smail for colortable & relative paths improvements)

from contextlib import contextmanager
import os
import math
import warnings
from xml.etree import ElementTree as ET
import rasterio
import numpy as np


from rasterio.dtypes import _gdal_typename
from rasterio.enums import MaskFlags
from rasterio.enums import Resampling
from rasterio._path import _parse_path
from rasterio.transform import Affine


def virtual_merge(
    datasets,
    bounds=None,
    res=None,
    nodata=None,
    background=None,
    hidenodata=False,
    dtype=None,
    indexes=None,
    output_count=None,
    resampling=Resampling.nearest,
    target_aligned_pixels=False,
    dst_path=None,
    dst_kwds=None,
) -> str:
    """Merge multiple datasets into a single virtual dataset.

    All files must have the same number of bands, data type, and
    coordinate reference system.

    Input files are merged in their listed order using the reverse
    painter's algorithm (default) or another method. If the output file
    exists, its values will be overwritten by input values.

    Geospatial bounds and resolution of a new output file in the units
    of the input file coordinate reference system may be provided and
    are otherwise taken from the first input file.

    Roughly equivalent to GDAL's GDALBuildVrt utility.

    Parameters
    ----------
    datasets : list
        List of dataset objects opened in 'r' mode, filenames or
        PathLike objects source datasets to be merged.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).  If not
        set, bounds are determined from bounds of input rasters.
    res: tuple, optional
        Output resolution in units of coordinate reference system. If
        not set, the resolution of the first raster is used. If a single
        value is passed, output pixels will be square.
    nodata: float, optional
        Nodata value to use in output file. If not set, uses the nodata
        value in the first input raster.
    background : int or float, optional
        The background fill value for the VRT.
    dtype: numpy.dtype or string
        Data type to use in outputfile. If not set, uses the dtype value
        in the first input raster.
    indexes : list of ints or a single int, optional
        Bands to read and merge
    output_count: int, optional
        If using callable it may be useful to have additional bands in
        the output in addition to the indexes specified for read
    resampling : Resampling, optional
        Resampling algorithm used when reading input files.  Default:
        `Resampling.nearest`.
    target_aligned_pixels : bool, optional
        Whether to adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the ``-tap``
        options of GDAL utilities.  Default: False.
    dst_path : str or PathLike, optional
        Path of output dataset
    dst_kwds : dict, optional
        Dictionary of creation options and other paramters that will be
        overlaid on the profile of the output dataset.

    Returns
    -------
    str
        XML text describing the virtual dataset (VRT).

    Notes
    -----
    Credits: https://github.com/rasterio/rasterio/pull/2699
    This function is based on a PR (pending for rasterio release 1.5.0).
    It has been slightly modified to allow the writing of relative paths.
    """

    if isinstance(datasets[0], (str, os.PathLike)):
        dataset_opener = rasterio.open
    else:

        @contextmanager
        def nullcontext(obj):
            try:
                yield obj
            finally:
                pass

        dataset_opener = nullcontext

    with dataset_opener(datasets[0]) as first:
        first_res = first.res
        nodataval = first.nodatavals[0]
        crs_wkt = first.crs.wkt
        indexes = first.indexes
        descriptions = first.descriptions
        colorinterps = first.colorinterp
        block_shapes = first.block_shapes
        dtypes = first.dtypes
        mask_flag_enums = first.mask_flag_enums
        dt = dtypes[0]

        if indexes is None:
            src_count = first.count
        elif isinstance(indexes, int):
            src_count = indexes
        else:
            src_count = len(indexes)

    if not output_count:
        output_count = src_count

    # Extent from option or extent of all inputs
    if bounds:
        dst_w, dst_s, dst_e, dst_n = bounds
    else:
        # scan input files
        xs = []
        ys = []
        for dataset in datasets:
            with dataset_opener(dataset) as src:
                left, bottom, right, top = src.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
        dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    # Resolution/pixel size
    if not res:
        res = first_res
    elif not np.iterable(res):
        res = (res, res)
    elif len(res) == 1:
        res = (res[0], res[0])

    if target_aligned_pixels:
        dst_w = math.floor(dst_w / res[0]) * res[0]
        dst_e = math.ceil(dst_e / res[0]) * res[0]
        dst_s = math.floor(dst_s / res[1]) * res[1]
        dst_n = math.ceil(dst_n / res[1]) * res[1]

    # Compute output shape. We guarantee it will cover the output bounds
    # completely.
    output_width = int(round((dst_e - dst_w) / res[0]))
    output_height = int(round((dst_n - dst_s) / res[1]))

    output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])

    if dtype is not None:
        dt = dtype

    # Create destination VRT.
    vrtdataset = ET.Element(
        "VRTDataset",
        rasterYSize=str(output_height),
        rasterXSize=str(output_width),
    )

    ET.SubElement(vrtdataset, "SRS").text = crs_wkt if crs_wkt else ""
    ET.SubElement(vrtdataset, "GeoTransform").text = ",".join(
        [str(v) for v in output_transform.to_gdal()]
    )

    if nodata is not None:
        # Only fill if the nodataval is within dtype's range
        inrange = False
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            inrange = info.min <= nodata <= info.max
        elif np.issubdtype(dt, np.floating):
            if math.isnan(nodata):
                inrange = True
            else:
                info = np.finfo(dt)
                inrange = info.min <= nodata <= info.max
        if inrange:
            nodataval = nodata
        else:
            warnings.warn(
                "The nodata value, %s, is beyond the valid "
                "range of the chosen data type, %s. Consider overriding it "
                "using the --nodata option for better results." % (nodataval, dt)
            )
    else:
        nodataval = None

    # Create VRT bands.
    for bidx, desc, ci, block_shape, dtype in zip(
        indexes, descriptions, colorinterps, block_shapes, dtypes
    ):
        vrtrasterband = ET.SubElement(
            vrtdataset,
            "VRTRasterBand",
            dataType=_gdal_typename(dtype),
            band=str(bidx),
        )

        if background is not None or nodataval is not None:
            ET.SubElement(vrtrasterband, "NoDataValue").text = str(
                background or nodataval
            )

            if hidenodata:
                ET.SubElement(vrtrasterband, "HideNoDataValue").text = "1"

        ET.SubElement(vrtrasterband, "ColorInterp").text = ci.name.capitalize()
        ET.SubElement(vrtrasterband, "Description").text = desc

    # Add sources to VRT bands.
    for idx, dataset in enumerate(datasets):
        with dataset_opener(dataset) as src:
            for bidx, ci, block_shape, dtype in zip(
                src.indexes, src.colorinterp, src.block_shapes, src.dtypes
            ):
                vrtrasterband = vrtdataset.find(f"VRTRasterBand[@band='{bidx}']")
                complexsource = ET.SubElement(
                    vrtrasterband, "ComplexSource", resampling=resampling.name
                )
                out_path, rel_to_vrt = format_paths(src.name, dst_path)
                ET.SubElement(
                    complexsource,
                    "SourceFilename",
                    relativeToVRT=rel_to_vrt,
                    shared="0",
                ).text = _parse_path(out_path).as_vsi()
                ET.SubElement(complexsource, "SourceBand").text = str(bidx)
                ET.SubElement(
                    complexsource,
                    "SourceProperties",
                    RasterXSize=str(output_width),
                    RasterYSize=str(output_height),
                    dataType=_gdal_typename(dtype),
                    BlockYSize=str(block_shape[0]),
                    BlockXSize=str(block_shape[1]),
                )
                ET.SubElement(
                    complexsource,
                    "SrcRect",
                    xOff="0",
                    yOff="0",
                    xSize=str(src.width),
                    ySize=str(src.height),
                )
                ET.SubElement(
                    complexsource,
                    "DstRect",
                    xOff=str(
                        (src.transform.xoff - output_transform.xoff)
                        / output_transform.a
                    ),
                    yOff=str(
                        (src.transform.yoff - output_transform.yoff)
                        / output_transform.e
                    ),
                    xSize=str(src.width * src.transform.a / output_transform.a),
                    ySize=str(src.height * src.transform.e / output_transform.e),
                )

                if src.nodata is not None:
                    ET.SubElement(complexsource, "NODATA").text = str(src.nodata)

                if src.options is not None:
                    openoptions = ET.SubElement(complexsource, "OpenOptions")
                    for ookey, oovalue in src.options.items():
                        ET.SubElement(openoptions, "OOI", key=str(ookey)).text = str(
                            oovalue
                        )

    if all(MaskFlags.per_dataset in flags for flags in mask_flag_enums):
        maskband = ET.SubElement(vrtdataset, "MaskBand")
        vrtrasterband = ET.SubElement(maskband, "VRTRasterBand", dataType="Byte")

        for idx, dataset in enumerate(datasets):
            with dataset_opener(dataset) as src:
                for bidx, ci, block_shape, dtype in zip(
                    src.indexes, src.colorinterp, src.block_shapes, src.dtypes
                ):
                    simplesource = ET.SubElement(
                        vrtrasterband, "SimpleSource", resampling=resampling.name
                    )
                    out_path, rel_to_vrt = format_paths(src.name, dst_path)
                    ET.SubElement(
                        simplesource,
                        "SourceFilename",
                        relativeToVRT=rel_to_vrt,
                        shared="0",
                    ).text = _parse_path(out_path).as_vsi()
                    ET.SubElement(simplesource, "SourceBand").text = "mask,1"
                    ET.SubElement(
                        simplesource,
                        "SourceProperties",
                        RasterXSize=str(output_width),
                        RasterYSize=str(output_height),
                        dataType="Byte",
                        BlockYSize=str(block_shape[0]),
                        BlockXSize=str(block_shape[1]),
                    )
                    ET.SubElement(
                        simplesource,
                        "SrcRect",
                        xOff="0",
                        yOff="0",
                        xSize=str(src.width),
                        ySize=str(src.height),
                    )
                    ET.SubElement(
                        simplesource,
                        "DstRect",
                        xOff=str(
                            (src.transform.xoff - output_transform.xoff)
                            / output_transform.a
                        ),
                        yOff=str(
                            (src.transform.yoff - output_transform.yoff)
                            / output_transform.e
                        ),
                        xSize=str(src.width * src.transform.a / output_transform.a),
                        ySize=str(src.height * src.transform.e / output_transform.e),
                    )

    if dst_path:
        doc = ET.tostring(vrtdataset).decode("UTF-8")
        with open(dst_path, "w") as vrt_file:
            vrt_file.write(doc)
    else:
        return ET.tostring(vrtdataset, encoding="utf-8")


def format_paths(src_path, dst_path=None):
    if dst_path:
        dst_path = os.path.abspath(dst_path)
        src_path = os.path.abspath(src_path)
        prefix = os.path.commonprefix([src_path, dst_path])
        if prefix:
            out_path = os.path.relpath(src_path, os.path.dirname(prefix))
            rel_path = "1"
        else:
            out_path = os.path.abspath(src_path)
            rel_path = "0"
    else:
        out_path = os.path.abspath(src_path)
        rel_path = "0"
    return out_path, rel_path
