"""
Tbd
* add logger for debugging
* introduce redundancy in case of failure
"""

import numpy as np
import os
import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import rasterio as rio
import time
import threading
import shutil
import uuid
import warnings
import xarray as xr

from copy import deepcopy
from itertools import product
from multiprocess import Pool, Manager
from rioxarray.merge import merge_arrays
from shapely.geometry import box
from tqdm import tqdm

import semantique as sq
from semantique import exceptions
from semantique.extent import SpatialExtent, TemporalExtent
from semantique.processor.arrays import Collection
from semantique.processor.core import QueryProcessor
from semantique.vrt import virtual_merge


class TileHandler:
    """Handler for executing a query in a (spatially or temporally) tiled manner.
    Currently only supports non-grouped outputs.

    Parameters
    ----------
      recipe : QueryRecipe
        The query recipe to be processed.
      datacube : Datacube
        The datacube instance to process the query against.
      space : SpatialExtent
        The spatial extent in which the query should be processed.
      time : TemporalExtent
        The temporal extent in which the query should be processed.
      spatial_resolution : :obj:`list`
        Spatial resolution of the grid. Should be given as a list in the format
        `[y, x]`, where y is the cell size along the y-axis, x is the cell size
        along the x-axis, and both are given as :obj:`int` or :obj:`float`
        value expressed in the units of the CRS. These values should include
        the direction of the axes. For most CRSs, the y-axis has a negative
        direction, and hence the cell size along the y-axis is given as a
        negative number.
      crs : optional
        Coordinate reference system in which the grid should be created. Can be
        given as any object understood by the initializer of
        :class:`pyproj.crs.CRS`. This includes :obj:`pyproj.crs.CRS` objects
        themselves, as well as EPSG codes and WKT strings. If :obj:`None`, the
        CRS of the extent itself is used.
      chunksize_t : int, tbd
        Temporal chunksize
      chunksize_s : int, tbd
        Spatial chunksize
      tile_dim : tbd
      merge : one of ["vrt", "single", None]
      out_dir : str
      caching : bool
      reauth : bool
      verbose : bool
      **config :
        Additional configuration parameters forwarded to QueryRecipe.execute.
        See :class:`QueryRecipe`, respectively :class:`QueryProcessor`.
        Needs to contain at least mapping instance to process the query against.
        (tbd: exclude mapping as obligatory parameter such that only optional
        parameters need to be placed here)
    """

    def __init__(
        self,
        recipe,
        datacube,
        space,
        time,
        spatial_resolution,
        crs=None,
        chunksize_t="1W",
        chunksize_s=1024,
        tile_dim=None,
        merge="single",
        out_dir=None,
        caching=False,
        reauth=True,
        verbose=False,
        **config,
    ):
        self.recipe = recipe
        self.datacube = datacube
        self.space = space
        self.time = time
        self.spatial_resolution = spatial_resolution
        self.crs = crs
        self.chunksize_t = chunksize_t
        self.chunksize_s = chunksize_s
        self.tile_dim = tile_dim
        self.merge = merge
        self.out_dir = out_dir
        self.caching = caching
        self.reauth = reauth
        self.verbose = verbose
        self.config = config
        self.setup()

    def setup(self):
        # init necessary attributes
        self.grid = None
        self.tile_results = []
        self.cache = None
        # retrieve crs information
        if not self.crs:
            self.crs = self.space.crs
        # retrieve tiling dimension
        self.get_tile_dim()
        # check merge-dependent prerequisites
        if self.merge == "vrt":
            pass
            # warnings.warn(
            #     "merge == vrt requires all outputs to have the same shape. "
            #     "Ensure that the recipe does not contain any trim operations. "
            #     "The datacube configuration will be set to trim=False."
            # )
            # self.datacube.config["trim"] = False
        elif self.merge == "vrt" and self.tile_dim == sq.dimensions.TIME:
            raise NotImplementedError(
                "If tiling is done along the temporal dimension, 'vrt' is "
                "currently not available as a merge strategy."
            )
        elif (self.merge == "vrt" or self.merge is None) and not self.out_dir:
            raise ValueError(
                f"An 'out_dir' argument must be provided when merge is set to {self.merge}."
            )
        # create output directory
        if self.out_dir:
            os.makedirs(self.out_dir)
        # continous re-auth
        if self.reauth:
            self.signing_thread_event = threading.Event()
            self.signing_thread = threading.Thread(target=self._continuous_signing)
            self.signing_thread.daemon = True
            self.signing_thread.start()

    def _continuous_signing(self):
        while not self.signing_thread_event.is_set():
            self.datacube.src = self.datacube._sign_metadata(list(self.datacube.src))
            time.sleep(1)

    # join thread when deleting the instance - not working yet
    def __del__(self):
        if self.signing_thread_event.is_set():
            self.signing_thread_event.set()
            self.signing_thread.join()

    def get_tile_dim(self):
        """Returns dimension usable for tiling & parallelisation of recipe execution.
        Calls `.get_op_dims()` to get dimensions which should be kept together to
        ensure safe tiling.

        Note: EO data is usually organised in a time-first file structure, i.e. each file
        contains many spatial observations for one point in time. Therefore, temporal chunking
        is favourable if possible. However, choosing a reasonable default for temporal chunksize
        is more difficult as the temporal spacing of EO obsevrations is unknown prior to data fetching.
        Therefore, chunking in space is set as a default if the processing chain as given by the
        recipe allows it.
        """
        reduce_dims = TileHandler.get_op_dims(self.recipe)
        # retrieve tile dimension as non-used dimension
        if reduce_dims is None:
            if not self.tile_dim:
                self.tile_dim = sq.dimensions.SPACE
        elif reduce_dims == [sq.dimensions.TIME]:
            if self.tile_dim:
                if self.tile_dim != sq.dimensions.SPACE:
                    warnings.warn(
                        f"Tiling dimension {self.tile_dim} will be overwritten. Tiling dimension is set to 'space'."
                    )
            self.tile_dim = sq.dimensions.SPACE
        elif reduce_dims == [sq.dimensions.SPACE]:
            if self.tile_dim:
                if self.tile_dim != sq.dimensions.TIME:
                    warnings.warn(
                        f"Tiling dimension {self.tile_dim} will be overwritten. Tiling dimension is set to 'time'."
                    )
            self.tile_dim = sq.dimensions.TIME
        else:
            warnings.warn("Tiling not feasible. Tiling dimension is set to 'None'.")
            self.tile_dim = None

    def get_tile_grid(self):
        """Creates spatial or temporal grid according to tiling dimension to enable
        subsequent sequential iteration over small sub-parts of the total extent object
        via .execute().
        """
        if self.tile_dim == sq.dimensions.TIME:
            # create temporal grid
            self.grid = self.create_temporal_grid(
                self.time["start"], self.time["end"], self.chunksize_t
            )

        elif self.tile_dim == sq.dimensions.SPACE:
            # create spatial grid
            if self.merge == "vrt" or self.merge is None:
                # tbd: allow both options of precise_shp here for both
                # set default for both to True (to ensure consistent results)
                # False only used for vrt to ensure same tile sizes (-> dl afterwards) & easier estimation of sizes
                # both things not really needed
                # precise_shp = True
                precise_shp = False
            else:
                precise_shp = True
            self.grid = self.create_spatial_grid(
                self.space,
                self.spatial_resolution,
                self.chunksize_s,
                self.crs,
                precise=precise_shp,
            )

    def execute(self):
        """Runs the QueryProcessor.execute() method for all tiles."""
        # A) dry-run is performed to set up cache
        self.preview()

        # B) eval recipe & postprocess in tile-wise manner
        for i, tile in tqdm(
            enumerate(self.grid), disable=not self.verbose, total=len(self.grid)
        ):
            # run workflow for single tile
            context = self._create_context(
                **{self.tile_dim: tile}, cache=deepcopy(self.cache)
            )
            _, response = TileHandler._execute_workflow(context)

            if response:
                # postprocess response
                if self.tile_dim == sq.dimensions.TIME:
                    response = self._postprocess_temporal(response)
                elif self.tile_dim == sq.dimensions.SPACE:
                    response = self._postprocess_spatial(response)
                # write result (in-memory or to disk)
                if self.merge == "single":
                    self.tile_results.append(response)
                elif self.merge == "vrt" or self.merge is None:
                    for layer in response.keys():
                        # write to disk
                        out_dir = os.path.join(self.out_dir, layer)
                        out_path = os.path.join(out_dir, f"{i}.tif")
                        os.makedirs(out_dir, exist_ok=True)
                        response[layer].rio.to_raster(out_path)
                        self.tile_results.append(out_path)
            # missing reponse when self.tile_dim = sq.dimensions.TIME & trim=True
            else:
                pass

        # C) optional merge of results
        if self.merge == "single":
            self.merge_single()
        elif self.merge == "vrt":
            self.merge_vrt()

    def merge_vrt(self):
        """Merges results obtained for individual tiles by creating a virtual raster.
        Only available for spatial results obtained by a reduce-over-time. Not implemented
        for temporal results (i.e. timeseries obtained by a reduce-over-space).
        """
        res_keys = [os.path.dirname(x).split(os.sep)[-1] for x in self.tile_results]
        for k in np.unique(res_keys):
            res_dir = os.path.join(self.out_dir, k)
            srcs = [os.path.join(res_dir, x) for x in os.listdir(res_dir)]
            # ensure same bands across tiles
            self._equalize_bands(srcs)
            # create virtual raster
            res_path = os.path.join(self.out_dir, f"{k}.vrt")
            virtual_merge(srcs, dst_path=res_path)
            # create overview for vrt
            dst = rio.open(res_path, "r+")
            vrt_scales = [4, 8, 16, 32, 64, 128, 256, 512]
            vrt_scales = [x for x in vrt_scales if x < max(dst.shape)]
            dst.build_overviews(vrt_scales)
            dst.update_tags(ns="rio_overview")
            dst.close()

    def _merge_temporal(src_arrs):
        """Merges temporally stratified results into an array"""
        if isinstance(src_arrs[0], xr.core.dataarray.DataArray):
            # merge across time
            dst_arr = xr.concat(src_arrs, dim=sq.dimensions.TIME)
        # this can be omitted?
        elif isinstance(src_arrs[0], Collection):
            dst_arrs = []
            # merge collection results
            for collection in src_arrs:
                grouper_vals = [x.name for x in collection]
                arr = xr.concat([x for x in collection], dim="grouper")
                arr = arr.assign_coords(grouper=grouper_vals)
                dst_arrs.append(arr)
            # merge across time
            dst_arr = xr.concat(dst_arrs, dim=sq.dimensions.TIME)
        else:
            raise NotImplementedError(f"No method for merging source array {src_arrs}.")
        return dst_arr

    def _merge_spatial(
        src_arrs,
        crs,
    ):
        """Merges spatially stratified results into an array"""
        # tbd: remove superfluous parts here
        # get dimensions of input array
        arr_dims = list(src_arrs[0].dims)
        # remove spatial dimensions
        if sq.dimensions.X in arr_dims:
            arr_dims.remove(sq.dimensions.X)
        if sq.dimensions.Y in arr_dims:
            arr_dims.remove(sq.dimensions.Y)
        # check if 2D/3D input arrays
        if len(arr_dims):
            # possibly remaining dimension is the temporal one (e.g. year, season, etc)
            # retrieve temporal values
            time_dim = arr_dims[0]
            time_vals = [x[time_dim] for x in src_arrs]
            time_vals = np.unique(xr.concat(time_vals, dim=time_dim).values)
            # for each timestep merge results spatially first
            arrs_main = []
            for time_val in time_vals:
                arrs_sub = []
                for arr in src_arrs:
                    # slice array for given timestep
                    try:
                        arr_slice = arr.sel(**{time_dim: time_val})
                    except KeyError:
                        continue
                    # introduce band coordinate as index variable
                    coords = {}
                    coords["band"] = (["band"], np.array([1]))
                    coords.update(
                        {dim: (dim, arr_slice[dim].values) for dim in arr_slice.dims}
                    )
                    new_arr = xr.DataArray(
                        data=np.expand_dims(arr_slice.values, 0),
                        coords=coords,
                        attrs=arr_slice.attrs,
                    )
                    new_arr = new_arr.rio.write_crs(crs)
                    arrs_sub.append(new_arr)
                # spatial merge
                merged_arr = merge_arrays(arrs_sub, crs=crs)
                merged_arr = merged_arr[0].drop_vars("band")
                # re-introducing time dimension
                coords = {}
                coords[time_dim] = ([time_dim], np.array([time_val]))
                coords.update(
                    {dim: (dim, merged_arr[dim].values) for dim in merged_arr.dims}
                )
                new_arr = xr.DataArray(
                    data=np.expand_dims(merged_arr.values, 0),
                    coords=coords,
                    attrs=merged_arr.attrs,
                )
                new_arr = new_arr.rio.write_crs(crs)
                arrs_main.append(new_arr)
            # merge across time
            joint_arr = xr.concat(arrs_main, dim=time_dim)
            # persist band names
            joint_arr.attrs["long_name"] = [str(x) for x in joint_arr[time_dim].values]
            joint_arr.attrs["band_variable"] = time_dim
        else:
            # direct spatial merge possible
            arrs = []
            for arr in src_arrs:
                # introduce band coordinate as index variable
                coords = {}
                coords["band"] = (["band"], np.array([1]))
                coords.update({dim: (dim, arr[dim].values) for dim in arr.dims})
                new_arr = xr.DataArray(
                    data=np.expand_dims(arr.values, 0),
                    coords=coords,
                    attrs=arr.attrs,
                )
                new_arr = new_arr.rio.write_crs(crs)
                arrs.append(new_arr)
            # spatial merge
            joint_arr = merge_arrays(arrs, crs=crs)
            joint_arr = joint_arr[0].drop_vars("band")
        return joint_arr

    def _postprocess_spatial(self, in_dict):
        """Postprocesses the response to ensure homogeneous response format,
        i.e. a dictionary containing xarrays with a at most 3 dimensions
        """
        out_dict = {}
        for k in in_dict.keys():
            in_arr = in_dict[k]
            # convert collections (grouped outputs) into arrays
            # problematic since grouper variable may exist already
            # needed for grouped outputs!
            if isinstance(in_arr, Collection):
                grouper_vals = [x.name for x in in_arr]
                if isinstance(grouper_vals[0], tuple):
                    grouper_vals = [str(x) for x in grouper_vals]
                in_arr = xr.concat([x for x in in_arr], dim="_grouper")
                in_arr = in_arr.assign_coords(_grouper=grouper_vals)
            # add crs information
            in_arr = in_arr.rio.write_crs(self.crs)
            # flatten 4D outputs to 3D
            re_dims = TileHandler._get_nonspatial_dims(in_arr)
            if len(re_dims) > 1:
                in_arr = in_arr.stack(grouper=re_dims)
                in_arr = in_arr.transpose("grouper", ...)
            # persist band names for 3D outputs
            re_dims = TileHandler._get_nonspatial_dims(in_arr)
            if len(re_dims):
                re_dim = re_dims[0]
                re_vals = [str(x) for x in in_arr[re_dim].values]
                in_arr.attrs["long_name"] = re_vals
                in_arr.attrs["band_variable"] = re_dim
            out_dict[k] = in_arr
        return out_dict

    def _postprocess_temporal(self, in_dict):
        """Postprocesses the response to ensure homogeneous response format"""
        out_dict = {}
        for k in in_dict.keys():
            in_arr = in_dict[k]
            # convert collections (grouped outputs) into arrays
            if isinstance(in_arr, Collection):
                grouper_vals = [x.name for x in in_arr]
                in_arr = xr.concat([x for x in in_arr], dim="_grouper")
                in_arr = in_arr.assign_coords(_grouper=grouper_vals)
            out_dict[k] = in_arr
        return out_dict

    def merge_single(self):
        """Merge results obtained for individual tiles by stitching them
        temporally or spatially depending on the tiling dimension.
        """
        joint_res = {}
        res_keys = self.tile_results[0].keys()
        # merge recipes results
        for k in res_keys:
            src_arrs = [x[k] for x in self.tile_results]
            if self.tile_dim == sq.dimensions.TIME:
                joint_arr = TileHandler._merge_temporal(src_arrs)
            elif self.tile_dim == sq.dimensions.SPACE:
                joint_arr = TileHandler._merge_spatial(src_arrs, self.crs)
            joint_arr.name = k
            joint_res[k] = joint_arr
        self.joint_res = joint_res
        # write to out_dir
        if self.out_dir:
            for k, v in self.joint_res.items():
                if self.tile_dim == sq.dimensions.TIME:
                    out_path = os.path.join(self.out_dir, f"{k}.nc")
                    v.to_netcdf(out_path)
                elif self.tile_dim == sq.dimensions.SPACE:
                    out_path = os.path.join(self.out_dir, f"{k}.tif")
                    v.rio.to_raster(out_path)

    def preview(self):
        """Estimator to see if results can be given as one final object
        or not.

        Works by retrieving the tile size and testing the script for one
        of the tiles.
        """
        # preview info
        time_info = (
            "preview() is currently only implemented for "
            "spatial outputs. Unless you are processing very dense timeseries "
            "and/or processing many features it's save to assume that the size "
            "of your output is rather small, so don't worry about the memory space.\n"
        )
        space_info = (
            "The following numbers are rough estimations depending on the chosen "
            "strategy for merging the individual tile results. If merge='single' "
            "is choosen the total size indicates a lower bound for how much RAM is "
            "required since the individual tile results will be stored there before "
            "merging.\n"
        )

        # get tiling grid
        if not self.grid:
            self.get_tile_grid()

        # preview run of workflow for a single tile
        tile_idx = 0
        valid_response = False
        while not valid_response:
            tile = self.grid[tile_idx]
            context = self._create_context(
                **{self.tile_dim: tile}, preview=True, cache=None
            )
            qp, response = TileHandler._execute_workflow(context)
            valid_response = True if response else False
            tile_idx += 1

        # postprocess response
        if self.tile_dim == sq.dimensions.TIME:
            response = self._postprocess_temporal(response)
        elif self.tile_dim == sq.dimensions.SPACE:
            response = self._postprocess_spatial(response)

        # init cache
        if self.caching:
            self.cache = qp.cache

        # get estimates based on preview run
        if self.tile_dim == sq.dimensions.TIME:
            print(time_info)
        elif self.tile_dim == sq.dimensions.SPACE:
            print(space_info)

            # retrieve amount of pixels for given spatial extent
            total_bbox = self.space._features.to_crs(self.crs).total_bounds
            width = total_bbox[2] - total_bbox[0]
            height = total_bbox[3] - total_bbox[1]
            num_pixels_x = int(np.ceil(width / abs(self.spatial_resolution[0])))
            num_pixels_y = int(np.ceil(height / abs(self.spatial_resolution[1])))
            xy_pixels = num_pixels_x * num_pixels_y

            # initialise dict to store layer information
            lyrs_info = {}
            for layer, arr in response.items():
                # compile general layer information
                lyr_info = {}
                lyr_info["dtype"] = arr.dtype
                lyr_info["res"] = self.spatial_resolution
                lyr_info["crs"] = self.crs
                # get array sizes (spatially & others)
                xy_dims = [sq.dimensions.X, sq.dimensions.Y]
                arr_xy_dims = [x for x in arr.dims if x in xy_dims]
                arr_z_dims = [x for x in arr.dims if x not in xy_dims]
                arr_xy = arr.isel(**{dim: 0 for dim in arr_z_dims})
                arr_z = arr.isel(**{dim: 0 for dim in arr_xy_dims})
                # extrapolate layer information for different merging strategies
                lyr_info["merge"] = {}
                # a) no merge
                scale = len(self.grid) * (self.chunksize_s**2) / arr_xy.size
                lyr_info["merge"]["None"] = {}
                lyr_info["merge"]["None"]["n"] = len(self.grid)
                lyr_info["merge"]["None"]["size"] = scale * arr.nbytes / (1024**3)
                lyr_info["merge"]["None"]["shape"] = (
                    *arr_z.shape,
                    self.chunksize_s,
                    self.chunksize_s,
                )
                # b) vrt
                vrt_scales = [4, 8, 16, 32, 64, 128, 256, 512]
                size_tiles = lyr_info["merge"]["None"]["size"]
                size_vrt = sum(
                    [
                        arr.nbytes * xy_pixels / arr_xy.size / (1024**3) / (x**2)
                        for x in vrt_scales
                    ]
                )
                lyr_info["merge"]["vrt"] = {}
                lyr_info["merge"]["vrt"]["n"] = len(self.grid)
                lyr_info["merge"]["vrt"]["size"] = size_tiles + size_vrt
                lyr_info["merge"]["vrt"]["shape"] = (
                    *arr_z.shape,
                    self.chunksize_s,
                    self.chunksize_s,
                )
                # c) merge into single array
                scale = xy_pixels / arr_xy.size
                lyr_info["merge"]["single"] = {}
                lyr_info["merge"]["single"]["n"] = 1
                lyr_info["merge"]["single"]["size"] = scale * arr.nbytes / (1024**3)
                lyr_info["merge"]["single"]["shape"] = (
                    *arr_z.shape,
                    num_pixels_x,
                    num_pixels_y,
                )
                lyrs_info[layer] = lyr_info

            # print general layer information
            max_l_lyr = max(len(r) for r in lyrs_info.keys())

            # part a) general information
            max_l_res = max([len(str(info["res"])) for lyr, info in lyrs_info.items()])
            line_l = max_l_lyr + max_l_res + 19
            print(line_l * "-")
            print("General layer info")
            print(line_l * "-")
            print(
                f"{'layer':{max_l_lyr}} : {'dtype':{9}} {'crs':{5}} {'res':{max_l_res}}"
            )
            print(line_l * "-")
            for lyr, info in lyrs_info.items():
                print(
                    f"{lyr:{max_l_lyr}} : {str(info['dtype']):{9}} {str(info['crs']):{5}} {str(info['res']):{max_l_res}}"
                )
            print(line_l * "-")
            print("")

            # part b) merge strategy dependend information
            for merge in lyrs_info[list(lyrs_info.keys())[0]]["merge"].keys():
                total_n = sum(
                    [info["merge"][merge]["n"] for info in lyrs_info.values()]
                )
                total_size = sum(
                    [info["merge"][merge]["size"] for info in lyrs_info.values()]
                )
                shapes = [
                    str(info["merge"][merge]["shape"]) for info in lyrs_info.values()
                ]
                max_l_n = len(f"{total_n}")
                max_l_size = len(f"{total_size:.2f}")
                max_l_shape = max([len(x) for x in shapes])
                line_l = max_l_lyr + max_l_n + max_l_size + max_l_shape + 18
                print(line_l * "-")
                print(f"Scenario: 'merge' = {merge}")
                print(line_l * "-")
                print(
                    f"{'layer':{max_l_lyr}} : {'size':^{max_l_size+3}}  {'tile n':^{max_l_n+8}}  {'tile shape':^{max_l_shape}}"
                )
                print(line_l * "-")
                for lyr, info in lyrs_info.items():
                    lyr_info = info["merge"][merge]
                    print(
                        f"{lyr:{max_l_lyr}} : {lyr_info['size']:>{max_l_size}.2f} Gb  {lyr_info['n']:>{max_l_n}} tile(s)  {str(lyr_info['shape']):>{max_l_shape}}"
                    )
                print(line_l * "-")
                print(
                    f"{'Total':{max_l_lyr}}   {total_size:{max_l_size}.2f} Gb  {total_n:{max_l_n}} tile(s)"
                )
                print(line_l * "-")
                print("")

    def _create_context(self, **kwargs):
        """Create execution context with dynamic space/time."""
        context = {
            "recipe": self.recipe,
            "datacube": self.datacube,
            "space": self.space,
            "time": self.time,
            "crs": self.crs,
            "spatial_resolution": self.spatial_resolution,
            **self.config,
        }
        context.update(kwargs)
        return context

    def _equalize_bands(self, src_paths):
        """Postprocesses the response to ensure avaliability of all bands"""
        # get non-spatial dims (i.e. band dimension)
        with rxr.open_rasterio(src_paths[0]) as src_arr:
            band_dims = TileHandler._get_nonspatial_dims(src_arr)
            band_dim = band_dims[0]
            n_bands = len(src_arr[band_dim])
        # ensure same bands across arrays
        if n_bands > 1:
            band_dim = band_dims[0]
            # retrieve values for non-spatial dim
            band_names = []
            for src in src_paths:
                with rxr.open_rasterio(src) as src_arr:
                    for band in src_arr.long_name:
                        band_names.append(band)
            band_names = list(np.unique(sorted(band_names)))
            # introduce missing values for single arrays
            for src in src_paths:
                with rxr.open_rasterio(src) as src_arr:
                    # create list to hold any new bands that need to be created
                    new_bands = []
                    for band in band_names:
                        if band not in src_arr.long_name:
                            # create an array of NaNs with the same shape as one band of src_arr
                            nan_band = np.full_like(
                                src_arr.isel(**{band_dim: 0}), np.nan
                            )
                            nan_band = np.expand_dims(nan_band, 0)
                            coords = {
                                c: (c, src_arr.coords[c].values) for c in src_arr.dims
                            }
                            coords.update({band_dim: (band_dim, np.array([0]))})
                            new_band = xr.DataArray(
                                nan_band, dims=src_arr.dims, coords=coords, name=band
                            )
                            new_band.attrs["long_name"] = band
                            new_bands.append(new_band)
                    # combine original bands with new, previosuly missing bands
                    if new_bands:
                        dst_arr = xr.concat(new_bands + [src_arr], dim=band_dim)
                        dst_arr.attrs["long_name"] = [
                            *[x.long_name for x in new_bands],
                            *src_arr.long_name,
                        ]
                    else:
                        dst_arr = src_arr
                    band_order = [dst_arr.long_name.index(x) for x in band_names]
                    dst_arr = dst_arr.isel(**{band_dim: band_order})
                    dst_arr.attrs["long_name"] = band_names
                    dst_arr[band_dim] = np.arange(len(band_names)) + 1
                # write updated array to disk
                TileHandler._write_to_origin(dst_arr, src)

    @staticmethod
    def _get_nonspatial_dims(in_arr):
        arr_dims = list(in_arr.dims)
        if sq.dimensions.X in arr_dims:
            arr_dims.remove(sq.dimensions.X)
        if sq.dimensions.Y in arr_dims:
            arr_dims.remove(sq.dimensions.Y)
        return arr_dims

    def _add_band_idx(self, in_arr):
        # introduce band coordinate as index variable
        coords = {}
        coords["band"] = (["band"], np.array([1]))
        coords.update({dim: (dim, in_arr[dim].values) for dim in in_arr.dims})
        out_arr = xr.DataArray(
            data=np.expand_dims(in_arr.values, 0),
            coords=coords,
            attrs=in_arr.attrs,
        )
        return out_arr

    @staticmethod
    def _execute_workflow(context):
        """Execute the workflow and handle response."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                qp = QueryProcessor.parse(**context)
                response = qp.optimize().execute()
                return qp, response
            except exceptions.EmptyDataError as e:
                warnings.warn(e)
                return None, None

    @staticmethod
    def get_op_dims(recipe_piece, dims=None):
        """Retrieves the dimensions over which operations take place.
        All operations indicated by verbs (such as reduce, groupby, etc) are considered.
        """
        if dims is None:
            dims = []
        if isinstance(recipe_piece, dict):
            # check if this dictionary matches the criteria
            if recipe_piece.get("type") == "verb":
                dim = recipe_piece.get("params").get("dimension")
                if dim:
                    dims.append(dim)
            # recursively search for values
            for value in recipe_piece.values():
                TileHandler.get_op_dims(value, dims)
        elif isinstance(recipe_piece, list):
            # if it's a list apply the function to each item in the list
            for item in recipe_piece:
                TileHandler.get_op_dims(item, dims)
        # categorise used dimensions into temporal & spatial dimensions
        dim_lut = {
            sq.dimensions.TIME: sq.dimensions.TIME,
            sq.dimensions.SPACE: sq.dimensions.SPACE,
        }
        dim_lut.update(
            {
                x: sq.dimensions.TIME
                for x in TileHandler._get_class_components(sq.components.time).values()
            }
        )
        dim_lut.update(
            {
                x: sq.dimensions.SPACE
                for x in TileHandler._get_class_components(sq.components.space).values()
            }
        )
        _dims = []
        for x in dims:
            try:
                _dims.append(dim_lut[x])
            except KeyError:
                pass
        dims = list(np.unique(_dims))
        return dims

    @staticmethod
    def create_temporal_grid(t_start, t_end, chunksize_t):
        time_grid = pd.date_range(t_start, t_end, freq=chunksize_t)
        time_grid = (
            [pd.Timestamp(t_start), *time_grid]
            if t_start not in time_grid
            else time_grid
        )
        time_grid = (
            [*time_grid, pd.Timestamp(t_end)] if t_end not in time_grid else time_grid
        )
        time_grid = [x for x in zip(time_grid, time_grid[1:])]
        time_grid = [TemporalExtent(*t) for t in time_grid]
        return time_grid

    @staticmethod
    def create_spatial_grid(
        space,
        spatial_resolution,
        chunksize_s,
        crs,
        precise=True,
    ):
        # create coarse spatial grid
        coarse_res = list(np.array(spatial_resolution) * chunksize_s)
        extent = space.rasterize(coarse_res, crs, all_touched=True)
        # get spatial spacings from coarse grid
        bounds = extent.rio.bounds()
        x_min, x_max = bounds[0], bounds[2]
        y_min, y_max = bounds[1], bounds[3]
        x_spacing = np.linspace(x_min, x_max, len(extent.x) + 1)
        y_spacing = np.linspace(y_min, y_max, len(extent.y) + 1)
        # construct sub-ranges
        _spatial_grid = list(
            product(
                [x for x in zip(x_spacing, x_spacing[1:])],
                [y for y in zip(y_spacing, y_spacing[1:])],
            )
        )
        _spatial_grid = [[x[0], y[0], x[1], y[1]] for x, y in _spatial_grid]
        # filter & mask tiles for shape geometry
        spatial_grid = []
        space = space.features.to_crs(extent.rio.crs)
        for tile in _spatial_grid:
            bbox_tile = box(*tile)
            bbox_tile = gpd.GeoDataFrame(geometry=[bbox_tile], crs=crs)
            if precise:
                tile_shape = bbox_tile.overlay(space, how="intersection").dissolve()
                if ((tile_shape.area / bbox_tile.area) > 0.0001).iloc[0]:
                    spatial_grid.append(SpatialExtent(tile_shape))
            else:
                if space.intersects(bbox_tile.unary_union).any():
                    tile_shape = bbox_tile.overlay(space, how="intersection").dissolve()
                    if ((tile_shape.area / bbox_tile.area) > 0.0001).iloc[0]:
                        spatial_grid.append(SpatialExtent(bbox_tile))
        return spatial_grid

    @staticmethod
    def _get_class_components(class_obj):
        """
        Function to get all components of the class along with their values
        """
        components = {}
        for attribute in dir(class_obj):
            if not attribute.startswith("__") and not callable(
                getattr(class_obj, attribute)
            ):
                components[attribute] = getattr(class_obj, attribute)
        return components

    @staticmethod
    def _write_to_origin(arr, path):
        """
        Write an opened rioxarray back to its original path,
        circumvents permission errors by temporarily writing to a new path
        & renaming afterwards
        """
        suffix = str(uuid.uuid4()).replace("-", "_")
        base, ext = os.path.splitext(path)
        temp_path = f"{base}_{suffix}_{ext}"
        arr.rio.to_raster(temp_path)
        shutil.move(temp_path, path)


class TileHandlerParallel(TileHandler):
    """Handler for executing a query in a tiled manner leveraging multiprocessing.
    The heavyweight and repetitive initialisation via preview() is carried out once and
    then tiles are processed in parallel.

    Note that for STACCubes, parallel processing is per default already enabled for data loading. Parallel processing via TileHandlerParallel therefore only makes sense if the workflow encapsulated in the recipe is significantly more time-consuming than the actual data loading. It must also be noted that the available RAM resources must be sufficient to process, n_procs times the amount of data that arises in the case of a simple TileHandler. This usually requires an adjustment of the chunksizes, which in turn may increase the amount of redundant data fetching processes (because the same data may be loaded for neighbouring smaller tiles). The possible advantage of using the ParallelProcessor therefore depends on the specific recipe and is not trivial. In case of doubt, the use of the TileHandler without multiprocessing is recommended.

    Note that the multiprocessing environment isolates the main process from the worker processes and ressources need to be serializable to be shared among worker processes. This implies that:
    A) Custom functions (verb, operators, reducers) need to be defined in a self-contained way, i.e. including imports such as `import semantique as sq` at their beginning.
    B) Reauth mechanisms relying on threaded processes won't work.
    """

    def __init__(self, *args, n_procs=os.cpu_count(), **kwargs):
        # threaded reauth is not serializable -> disable
        kwargs["reauth"] = False
        super().__init__(*args, **kwargs)
        self.n_procs = n_procs
        self.preview()

    def execute(self):
        # get grid idxs
        grid_idxs = np.arange(len(self.grid))
        # use manager to create a proxy for self that can be shared across processes
        with Manager() as manager:
            shared_self = manager.dict()
            shared_self.update({"instance": self})
            # run individual processes in parallelised manner
            with Pool(processes=self.n_procs) as pool:
                func = lambda idx: self._execute_tile(idx, shared_self)
                tile_results = list(
                    tqdm(pool.imap(func, grid_idxs), total=len(grid_idxs))
                )
        # flatten results
        self.tile_results = [x for sl in tile_results for x in sl]
        # merge results
        if self.merge == "single":
            self.merge_single()
        elif self.merge == "vrt":
            self.merge_vrt()

    def _execute_tile(self, tile_idx, shared_self):
        # get shared instance
        th = shared_self["instance"]
        # set data loading to single processes
        dc = th.datacube
        dc.config["dask_params"] = {"scheduler": "single-threaded"}
        # create context for specific tile
        context_params = {
            **{th.tile_dim: th.grid[tile_idx]},
            "cache": th.cache,
            "datacube": dc,
        }
        context = th._create_context(**context_params)
        # evaluate the recipe
        _, response = th._execute_workflow(context)
        # handle response
        if response:
            # write result (in-memory or to disk)
            if th.merge == "single":
                out = list(response)
            elif th.merge == "vrt" or th.merge is None:
                out = []
                for layer in response:
                    # write to disk
                    out_dir = os.path.join(th.out_dir, layer)
                    out_path = os.path.join(out_dir, f"{tile_idx}.tif")
                    os.makedirs(out_dir, exist_ok=True)
                    layer = response[layer].rio.write_crs(th.crs)
                    layer.rio.to_raster(out_path)
                    out.append(out_path)
            return out
