import numpy as np
import geopandas as gpd
import pandas as pd
import warnings
import xarray as xr

from itertools import product
from rioxarray.merge import merge_arrays
from shapely.geometry import box
from tqdm import tqdm

from semantique import exceptions
from semantique.extent import SpatialExtent, TemporalExtent
from semantique.processor.utils import parse_extent


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
        # check output options
        if self.merge == "direct":
            self.tile_results = []
        elif self.merge == "vrt" and not self.out_dir:
            raise ValueError(
                "An 'out_dir' argument must be provided when merge is set to 'vrt'."
            )

    def get_op_dims(self, recipe_piece, results=None):
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
                self.get_op_dims(value, results)
        elif isinstance(recipe_piece, list):
            # if it's a list apply the function to each item in the list
            for item in recipe_piece:
                self.get_op_dims(item, results)
        return list(np.unique(results))

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
        reduce_dims = self.get_op_dims(self.recipe)
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
        via .excute().
        """
        if self.tile_dim == "time":
            # create temporal grid
            t_start = self.time["start"]
            t_end = self.time["end"]
            time_grid = pd.date_range(t_start, t_end, freq=self.chunksize_t)
            time_grid = (
                [pd.Timestamp(t_start), *time_grid]
                if t_start not in time_grid
                else time_grid
            )
            time_grid = (
                [*time_grid, pd.Timestamp(t_end)]
                if t_end not in time_grid
                else time_grid
            )
            time_grid = [x for x in zip(time_grid, time_grid[1:])]
            time_grid = [TemporalExtent(*t) for t in time_grid]
            print(f"Dividing into {len(time_grid)} timesteps.")
            self.grid = time_grid

        elif self.tile_dim == "space":
            # create coarse spatial grid
            coarse_res = list(np.array(self.spatial_resolution) * self.chunksize_s)
            extent = self.space.rasterize(coarse_res, self.crs, all_touched=True)
            if not self.crs:
                self.crs = extent.rio.crs.to_epsg()
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
            space = self.space.features.to_crs(extent.rio.crs)
            for tile in _spatial_grid:
                bbox_tile = box(*tile)
                bbox_tile = gpd.GeoDataFrame(geometry=[bbox_tile], crs=self.crs)
                tile_shape = bbox_tile.overlay(space, how="intersection")
                if len(tile_shape):
                    spatial_grid.append(SpatialExtent(tile_shape))
            print(f"Dividing into {len(spatial_grid)} tiles.")
            self.grid = spatial_grid

    def execute(self):
        """Runs the QueryProcessor.execute() method for all tiles."""
        if self.tile_dim == "time":
            for time in tqdm(self.grid, disable=not self.verbose):
                # retrieve temporal subset & define context
                context = {
                    "datacube": self.datacube,
                    "space": self.space,
                    "time": time,
                    "crs": self.crs,
                    "spatial_resolution": self.spatial_resolution,
                    **self.config,
                }
                # run workflow
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        response = self.recipe.execute(**context)
                    except exceptions.EmptyDataError as e:
                        warnings.warn(e)
                # handle response
                self.tile_results.append(response)

        elif self.tile_dim == "space":
            for space in tqdm(self.grid, disable=not self.verbose):
                # # retrieve spatial subset
                # bbox_tile = box(*tile)
                # bbox_tile = gpd.GeoDataFrame(geometry=[bbox_tile], crs=self.crs)
                # space = SpatialExtent(bbox_tile)
                # run workflow
                context = {
                    "datacube": self.datacube,
                    "space": space,
                    "time": self.time,
                    "crs": self.crs,
                    "spatial_resolution": self.spatial_resolution,
                    **self.config,
                }
                # run workflow
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        response = self.recipe.execute(**context)
                    except exceptions.EmptyDataError as e:
                        warnings.warn(e)
                # handle response
                self.tile_results.append(response)

    def merge_results(self):
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
                joint_res[k] = xr.concat(src_arr, dim="time")
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

    def estimate_size(self):
        """Estimator to see if results can be given as one final object
        or not.
        """
        pass

    def create_vrt(self):
        """Build vrt based on tiles."""
        pass
