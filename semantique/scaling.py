"""
Tbd
* add logger for debugging
* introduce redundancy in case of failure
"""

import numpy as np
import os
import geopandas as gpd
import pandas as pd
import threading
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
            warnings.warn(
                "merge == vrt requires all outputs to have the same shape. "
                "Ensure that the recipe does not contain any trim operations. "
                "The datacube configuration will be set to trim=False."
            )
            self.datacube.config["trim"] = False
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
        # start re-auth
        thread = threading.Thread(target=self.datacube._sign_metadata)
        thread.daemon = True
        thread.start()

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
        # dry-run is required
        # read cache and write to object to read from for all subsequent calls
        if self.tile_dim == sq.dimensions.TIME:
            self.get_tile_grid()
            # preview run of workflow for a single tile
            tile = self.grid[0]
            context = self._create_context(
                **{self.tile_dim: tile}, preview=True, cache=None
            )
            qp, response = TileHandler._execute_workflow(context)
            # init cache
            if self.caching:
                self.cache = qp.cache
        elif self.tile_dim == sq.dimensions.SPACE:
            self.estimate_size()

        for i, tile in tqdm(
            enumerate(self.grid), disable=not self.verbose, total=len(self.grid)
        ):
            # run workflow for single tile
            context = self._create_context(
                **{self.tile_dim: tile}, cache=deepcopy(self.cache)
            )
            _, response = TileHandler._execute_workflow(context)

            # handle missing reponse
            # possible in cases where self.tile_dim = sq.dimensions.TIME & trim=True
            if response:
                # write result (in-memory or to disk)
                if self.merge == "single":
                    self.tile_results.append(response)
                elif self.merge == "vrt" or self.merge is None:
                    for layer in response:
                        # write to disk
                        out_dir = os.path.join(self.out_dir, layer)
                        out_path = os.path.join(out_dir, f"{i}.tif")
                        os.makedirs(out_dir, exist_ok=True)
                        layer = response[layer].rio.write_crs(self.crs)
                        layer.rio.to_raster(out_path)
                        self.tile_results.append(out_path)

        # merge results
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
            res_path = os.path.join(self.out_dir, f"{k}.vrt")
            virtual_merge(srcs, dst_path=res_path)

    def merge_single(self):
        """Merge results obtained for individual tiles by stitching them
        temporally or spatially depending on the tiling dimension.
        """
        joint_res = {}
        res_keys = self.tile_results[0].keys()
        for k in res_keys:
            # retrieve partial results
            src_arr = [x[k] for x in self.tile_results]
            # if temporal result
            if self.tile_dim == sq.dimensions.TIME:
                if isinstance(src_arr[0], xr.core.dataarray.DataArray):
                    joint_res[k] = xr.concat(src_arr, dim=sq.dimensions.TIME)
                elif isinstance(src_arr[0], Collection):
                    arrs = []
                    # merge collection results
                    for collection in src_arr:
                        grouper_vals = [x.name for x in collection]
                        arr = xr.concat([x for x in collection], dim="grouper")
                        arr = arr.assign_coords(grouper=grouper_vals)
                        arrs.append(arr)
                    # merge across time
                    joint_arr = xr.concat(arrs, dim=sq.dimensions.TIME)
                    joint_arr.name = k
                    joint_res[k] = joint_arr
                else:
                    raise NotImplementedError(
                        f"No method for merging source array {src_arr}."
                    )
            # if spatial result
            elif self.tile_dim == sq.dimensions.SPACE:
                # get dimensions of input array
                arr_dims = list(src_arr[0].dims)
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
                    time_vals = [x[time_dim] for x in src_arr]
                    time_vals = np.unique(xr.concat(time_vals, dim=time_dim).values)
                    # for each timestep merge results spatially first
                    arrs_main = []
                    for time_val in time_vals:
                        arrs_sub = []
                        for arr in src_arr:
                            # slice array for given timestep
                            try:
                                arr_slice = arr.sel(**{time_dim: time_val})
                            except KeyError:
                                continue
                            # introduce band coordinate as index variable
                            coords = {}
                            coords["band"] = (["band"], np.array([1]))
                            coords.update(
                                {
                                    dim: (dim, arr_slice[dim].values)
                                    for dim in arr_slice.dims
                                }
                            )
                            new_arr = xr.DataArray(
                                data=np.expand_dims(arr_slice.values, 0),
                                coords=coords,
                                attrs=arr_slice.attrs,
                            )
                            new_arr = new_arr.rio.write_crs(self.crs)
                            arrs_sub.append(new_arr)
                        # spatial merge
                        merged_arr = merge_arrays(arrs_sub, crs=self.crs)
                        merged_arr = merged_arr[0].drop_vars("band")
                        # re-introducing time dimension
                        coords = {}
                        coords[time_dim] = ([time_dim], np.array([time_val]))
                        coords.update(
                            {
                                dim: (dim, merged_arr[dim].values)
                                for dim in merged_arr.dims
                            }
                        )
                        new_arr = xr.DataArray(
                            data=np.expand_dims(merged_arr.values, 0),
                            coords=coords,
                            attrs=merged_arr.attrs,
                        )
                        new_arr = new_arr.rio.write_crs(self.crs)
                        arrs_main.append(new_arr)
                    # merge across time
                    joint_arr = xr.concat(arrs_main, dim=time_dim)
                    # persist band names
                    joint_arr.attrs["long_name"] = [
                        str(x) for x in joint_arr[time_dim].values
                    ]
                    joint_arr.attrs["band_variable"] = time_dim
                else:
                    # direct spatial merge possible
                    arrs = []
                    for arr in src_arr:
                        # introduce band coordinate as index variable
                        coords = {}
                        coords["band"] = (["band"], np.array([1]))
                        coords.update({dim: (dim, arr[dim].values) for dim in arr.dims})
                        new_arr = xr.DataArray(
                            data=np.expand_dims(arr.values, 0),
                            coords=coords,
                            attrs=arr.attrs,
                        )
                        new_arr = new_arr.rio.write_crs(self.crs)
                        arrs.append(new_arr)
                    # spatial merge
                    joint_arr = merge_arrays(arrs, crs=self.crs)
                    joint_arr = joint_arr[0].drop_vars("band")
                # rename & write to overall dict
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

    def estimate_size(self):
        """Estimator to see if results can be given as one final object
        or not.

        Works by retrieving the tile size and testing the script for one
        of the tiles.
        """
        # preliminary checks
        time_info = (
            "Estimate_size() is currently only implemented for "
            "spatial outputs. Unless you are processing very dense timeseries "
            "and/or processing many features it's save to assume that the size "
            "of your output is rather small, so don't worry about the memory space.\n"
        )
        space_info = (
            "The following numbers are rough estimations depending on the chosen "
            "strategy for merging the individual tile results. If merge='single' "
            "is choosen the numbers indicate a lower bound for how much RAM is required "
            "since the individual tile results will be stored there before merging.\n"
        )
        if self.tile_dim == sq.dimensions.TIME:
            raise NotImplementedError(time_info)
        elif self.tile_dim == sq.dimensions.SPACE:
            print(space_info)
        if not self.grid:
            self.get_tile_grid()

        # preview run of workflow for a single tile
        tile = self.grid[0]
        context = self._create_context(
            **{self.tile_dim: tile}, preview=True, cache=None
        )
        qp, response = TileHandler._execute_workflow(context)

        # init cache
        if self.caching:
            self.cache = qp.cache

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
            # a) merge into single array
            scale = xy_pixels / arr_xy.size
            lyr_info["merge"]["single"] = {}
            lyr_info["merge"]["single"]["strategies"] = ["single"]
            lyr_info["merge"]["single"]["n"] = 1
            lyr_info["merge"]["single"]["size"] = scale * arr.nbytes / (1024**3)
            lyr_info["merge"]["single"]["shape"] = (
                *arr_z.shape,
                num_pixels_x,
                num_pixels_y,
            )
            # b) vrt or no merge
            scale = len(self.grid) * (self.chunksize_s**2) / arr_xy.size
            lyr_info["merge"]["multiple"] = {}
            lyr_info["merge"]["multiple"]["strategies"] = ["vrt", None]
            lyr_info["merge"]["multiple"]["n"] = len(self.grid)
            lyr_info["merge"]["multiple"]["size"] = scale * arr.nbytes / (1024**3)
            lyr_info["merge"]["multiple"]["shape"] = (
                *arr_z.shape,
                self.chunksize_s,
                self.chunksize_s,
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
        print(f"{'layer':{max_l_lyr}} : {'dtype':{9}} {'crs':{5}} {'res':{max_l_res}}")
        print(line_l * "-")
        for lyr, info in lyrs_info.items():
            print(
                f"{lyr:{max_l_lyr}} : {str(info['dtype']):{9}} {str(info['crs']):{5}} {str(info['res']):{max_l_res}}"
            )
        print(line_l * "-")
        print("")

        # part b) merge strategy dependend information
        for merge in lyrs_info[list(lyrs_info.keys())[0]]["merge"].keys():
            mtypes = lyrs_info[list(lyrs_info.keys())[0]]["merge"][merge]["strategies"]
            total_n = sum([info["merge"][merge]["n"] for info in lyrs_info.values()])
            total_size = sum(
                [info["merge"][merge]["size"] for info in lyrs_info.values()]
            )
            max_shape = max(
                [info["merge"][merge]["shape"] for info in lyrs_info.values()]
            )
            max_l_n = len(f"{total_n}")
            max_l_size = len(f"{total_size:.2f}")
            max_l_shape = len(str(max_shape))
            line_l = max_l_lyr + max_l_n + max_l_size + max_l_shape + 18
            print(line_l * "-")
            print(f"Scenario: 'merge' = {' or '.join([str(x) for x in mtypes])}")
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
        dims = list(np.unique([dim_lut[x] for x in dims]))
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
    def create_spatial_grid(space, spatial_resolution, chunksize_s, crs, precise=True):
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
                tile_shape = bbox_tile.overlay(space, how="intersection")
                if len(tile_shape):
                    spatial_grid.append(SpatialExtent(tile_shape))
            else:
                if space.intersects(bbox_tile.unary_union).iloc[0]:
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


class TileHandlerParallel(TileHandler):
    """Handler for executing a query in a tiled manner leveraging multiprocessing.
    The heavyweight and repetitive initialisation via estimate_size() is carried out once and
    then tiles are processed in parallel.

    Note that for STACCubes, parallel processing is per default already enabled for data loading. Parallel processing via TileHandlerParallel therefore only makes sense if the workflow encapsulated in the recipe is significantly more time-consuming than the actual data loading. It must also be noted that the available RAM resources must be sufficient to process, n_procs times the amount of data that arises in the case of a simple TileHandler. This usually requires an adjustment of the chunksizes, which in turn may increase the amount of redundant data fetching processes (because the same data may be loaded for neighbouring smaller tiles). The possible advantage of using the ParallelProcessor therefore depends on the specific recipe and is not trivial. In case of doubt, the use of the TileHandler without multiprocessing is recommended.

    Note that custom functions (verb, operators, reducers) need to be defined in a self-contained way,
    i.e. including imports such as `import semantique as sq` at their beginning since the
    multiprocessing environment isolates the main process from the worker processes and the function is not serializable.
    """

    def __init__(self, *args, n_procs=os.cpu_count(), **kwargs):
        super().__init__(*args, **kwargs)
        self.n_procs = n_procs
        self.estimate_size()

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
