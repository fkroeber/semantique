import numpy as np
import os
import geopandas as gpd
import pandas as pd
import warnings
import xarray as xr

from itertools import product
from multiprocess import Pool
from rioxarray.merge import merge_arrays
from shapely.geometry import box
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import semantique as sq
from semantique import exceptions
from semantique.extent import SpatialExtent, TemporalExtent
from semantique.processor.arrays import Collection
from semantique.vrt import virtual_merge


class TileHandler:
    """Handler for executing a query in a (spatially or temporally) tiled manner.

    Parameters
    ----------
      recipe : QueryRecipe
        The query recipe to be processed.
      datacube : Datacube
        The datacube instance to process the query against.
      space : tbd
      time : tbd
      chunksize_t : tbd
      chunksize_s : tbd
      merge : tbd
      **config :
        Additional configuration parameters forwarded to QueryRecipe.execute.
        See :class:`QueryRecipe`, respectively :class:`QueryProcessor`.
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
        merge="direct",
        out_dir=None,
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
        self.verbose = verbose
        self.config = config
        self.setup()

    def setup(self):
        # init necessary attributes
        self.grid = None
        self.tile_results = []
        # retrieve crs information
        if not self.crs:
            self.crs = self.space.crs
        # retrieve tiling dimension
        self.get_tile_dim()
        # check output options
        if self.merge == "vrt" and self.tile_dim == "time":
            raise NotImplementedError(
                "If tiling is done along the temporal dimension, only 'direct' is "
                "available as a merge strategy for now."
            )
        elif self.merge == "vrt" and not self.out_dir:
            raise ValueError(
                "An 'out_dir' argument must be provided when merge is set to 'vrt'."
            )
        elif self.merge == "vrt" and self.out_dir:
            os.makedirs(self.out_dir)

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
                self.tile_dim = "space"
        elif reduce_dims == ["time"]:
            if self.tile_dim:
                if self.tile_dim != "space":
                    warnings.warn(
                        f"Tiling dimension {self.tile_dim} will be overwritten. Tiling dimension is set to 'space'."
                    )
            self.tile_dim = "space"
        elif reduce_dims == ["space"]:
            if self.tile_dim:
                if self.tile_dim != "time":
                    warnings.warn(
                        f"Tiling dimension {self.tile_dim} will be overwritten. Tiling dimension is set to 'time'."
                    )
            self.tile_dim = "time"
        else:
            warnings.warn("Tiling not feasible. Tiling dimension is set to 'None'.")
            self.tile_dim = None

    def get_tile_grid(self):
        """Creates spatial or temporal grid according to tiling dimension to enable
        subsequent sequential iteration over small sub-parts of the total extent object
        via .execute().
        """
        if self.tile_dim == "time":
            # create temporal grid
            self.grid = self.create_temporal_grid(
                self.time["start"], self.time["end"], self.chunksize_t
            )

        elif self.tile_dim == "space":
            # create spatial grid
            self.grid = self.create_spatial_grid(
                self.space,
                self.spatial_resolution,
                self.chunksize_s,
                self.crs,
                precise=False,
            )

    def execute(self):
        """Runs the QueryProcessor.execute() method for all tiles."""
        # preliminaries
        if not self.grid:
            self.get_tile_grid()

        for i, tile in tqdm(
            enumerate(self.grid), disable=not self.verbose, total=len(self.grid)
        ):
            # run workflow for single tile
            context = self._create_context(**{self.tile_dim: tile})
            response = self._execute_workflow(context)

            # handle reponse
            # note: no response may occur in cases where self.tile_dim=="time"
            if response:
                # write result (in-memory or to disk)
                if self.merge == "direct":
                    self.tile_results.append(response)
                elif self.merge == "vrt":
                    for layer in response:
                        # write to disk
                        out_dir = os.path.join(self.out_dir, layer)
                        out_path = os.path.join(out_dir, f"{i}.tif")
                        os.makedirs(out_dir, exist_ok=True)
                        layer = response[layer].rio.write_crs(self.crs)
                        layer.rio.to_raster(out_path)
                        self.tile_results.append(out_path)

        # merge results
        if self.merge == "direct":
            self.merge_direct()
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
            res_path = os.path.join(self.out_dir, f"{k}.vrt")
            srcs = [os.path.join(res_dir, x) for x in os.listdir(res_dir)]
            virtual_merge(srcs, dst_path=res_path)

    def merge_direct(self):
        """Merge results obtained for individual tiles by stitching them
        temporally or spatially depending on the tiling dimension.
        """
        joint_res = {}
        res_keys = self.tile_results[0].keys()
        for k in res_keys:
            # retrieve partial results
            src_arr = [x[k] for x in self.tile_results]
            # if temporal result
            if self.tile_dim == "time":
                if isinstance(src_arr[0], xr.core.dataarray.DataArray):
                    joint_res[k] = xr.concat(src_arr, dim="time")
                elif isinstance(src_arr[0], Collection):
                    arrs = []
                    # merge collection results
                    for collection in src_arr:
                        grouper_vals = [x.name for x in collection]
                        arr = xr.concat([x for x in collection], dim="grouper")
                        arr = arr.assign_coords(grouper=grouper_vals)
                        arrs.append(arr)
                    # merge across time
                    joint_arr = xr.concat(arrs, dim="time")
                    joint_arr.name = k
                    joint_res[k] = joint_arr
                else:
                    raise ValueError("Not implemented.")
            # if spatial result
            elif self.tile_dim == "space":
                arrs = []
                for arr in src_arr:
                    new_arr = xr.DataArray(
                        data=np.expand_dims(arr.values, 0),
                        coords=dict(
                            band=(["band"], np.array([1])),
                            y=(["y"], arr["y"].values),
                            x=(["x"], arr["x"].values),
                        ),
                        attrs=arr.attrs,
                    ).rio.write_crs(self.crs)
                    arrs.append(new_arr)
                merged_arr = merge_arrays(arrs, crs=self.crs)
                merged_arr = merged_arr[0].drop_vars("band")
                joint_res[k] = merged_arr
        self.joint_res = joint_res
        # write to out_dir
        if self.out_dir:
            for k, v in self.joint_res.items():
                out_path = os.path.join(self.joint_res, f"{k}.tif")
                self.joint_res[k].rio.to_raster(out_path)

    def estimate_size(self):
        """Estimator to see if results can be given as one final object
        or not.

        Works by retrieving the tile size and testing the script for one
        of the tiles. (to be refined)
        """
        # preliminary checks
        time_info = (
            "The output_preview function is currently only implemented for "
            "spatial outputs. Unless you are processing very dense timeseries "
            "and/or processing many features it's save to assume that the size "
            "of your output is rather small, so don't worry about the memory space.\n"
        )
        space_info = (
            "The following numbers are rough estimations depending on the chosen "
            "strategy for merging the individual tile results. If merge='direct' "
            "is choosen the numbers indicate a lower bound for how much RAM is required "
            "since the individual tile results will be stored there before merging.\n"
        )
        if self.tile_dim == "time":
            raise NotImplementedError(time_info)
        elif self.tile_dim == "space":
            print(space_info)
        if not self.grid:
            self.get_tile_grid()

        # preview run of workflow for a single tile
        tile = self.grid[0]
        context = self._create_context(**{self.tile_dim: tile}, preview=True)
        response = self._execute_workflow(context)

        # retrieve amount of pixels for given spatial extent
        total_bbox = self.space._features.to_crs(self.crs).total_bounds
        width = total_bbox[2] - total_bbox[0]
        height = total_bbox[3] - total_bbox[1]
        num_pixels_x = int(np.ceil(width / abs(self.spatial_resolution[0])))
        num_pixels_y = int(np.ceil(height / abs(self.spatial_resolution[1])))
        total_pixels = num_pixels_x * num_pixels_y

        first_k = list(response.keys())[0]
        scale_f1 = total_pixels / response[first_k].size
        scale_f2 = len(self.grid) * (self.chunksize_s**2) / response[first_k].size
        merge_dict = {"merge (direct)": scale_f1, "merge (vrt)": scale_f2}

        # return scaled numbers for each response
        max_length = max(len(l) for l in response) + 1
        print("----------------------------------------")
        print("General layer info")
        print("----------------------------------------")
        for layer in response:
            dtype = response[layer].dtype
            print(f"{layer:<{max_length}}: {dtype}")
        print("")

        for k, s in merge_dict.items():
            max_length = max(len(lyr) for lyr in response)
            max_length = max(max_length, len("Total size")) + 3
            total_size = sum(
                (s * response[lyr].nbytes / (1024**3)) for lyr in response
            )
            print("----------------------------------------")
            print(f"Scenario: {k}")
            print("----------------------------------------")
            print(f"{'Total size':<{max_length}}: {total_size:.2f} Gb")
            for layer in response:
                size_gb = s * response[layer].nbytes / (1024**3)
                print(f"* {layer:<{max_length - 2}}: {size_gb:.2f} Gb")
            print("")
            if k == "merge (direct)":
                shape = f"({num_pixels_x}, {num_pixels_y})"
                n_tiles = 1
            elif k == "merge (vrt)":
                shape = f"({self.chunksize_s}, {self.chunksize_s})"
                n_tiles = len(self.grid)
            resolution = self.spatial_resolution
            crs = self.crs
            print(f"{'number of tiles':<{16}}: {n_tiles}")
            print(f"{'shape':<{16}}: {shape}")
            print(f"{'resolution':<{16}}: {resolution}")
            print(f"{'crs':<{16}}: {crs}")
            print("")

    def _create_context(self, **kwargs):
        """Create execution context with dynamic space/time."""
        context = {
            "datacube": self.datacube,
            "space": self.space,
            "time": self.time,
            "crs": self.crs,
            "spatial_resolution": self.spatial_resolution,
            **self.config,
        }
        context.update(kwargs)
        return context

    def _execute_workflow(self, context):
        """Execute the workflow and handle response."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                response = self.recipe.execute(**context)
                return response
            except exceptions.EmptyDataError as e:
                warnings.warn(e)

    @staticmethod
    def get_op_dims(recipe_piece, results=None):
        """Retrieves the dimensions over which operations take place.
        All operations indicated by verbs (such as reduce, groupby, etc) are considered.
        """
        if results is None:
            results = []
        if isinstance(recipe_piece, dict):
            # check if this dictionary matches the criteria
            if recipe_piece.get("type") == "verb":
                dim = recipe_piece.get("params").get("dimension")
                if dim:
                    results.append(dim)
            # recursively search for values
            for value in recipe_piece.values():
                TileHandler.get_op_dims(value, results)
        elif isinstance(recipe_piece, list):
            # if it's a list apply the function to each item in the list
            for item in recipe_piece:
                TileHandler.get_op_dims(item, results)
        return list(np.unique(results))

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
