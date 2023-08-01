import numpy as np
import xarray as xr
from hecuba import StorageDict
from hecuba.storageobj import StorageObj


class Coordinates(StorageDict):
    '''
       @TypeSpec dict<<key:str>,values:ndarray>
    '''
    pass


## coords are usually a dataarray themselves, and it allows
## representing complicated datasets
## this is same as datavar, use that instead?
class Coordinatevars(StorageDict):
    '''
    @TypeSpec dict<<var_name:str>,values:models.DataArray>
    '''


class DataArray(StorageObj):
    '''
    @ClassField dims list<str>
    @ClassField coords models.Coordinates
    @ClassField values ndarray
    '''

    @classmethod
    def from_xr_dataarray(cls, xr_dataarray):
        inst = cls()
        # if not dataarray raise
        inst.dims = list(xr_dataarray.dims)
        for coord_name, coord_val in xr_dataarray.coords.items():
            if coord_name in xr_dataarray.dims:  ## temp fix to assign arrays otherwise would need to
                # change coordinate model to have dataarray
                inst.coords[coord_name] = coord_val.values
        inst.values = xr_dataarray.values
        return inst

    @classmethod
    def to_xr_dataarray(cls, dataarray_alias):
        return xr_dataarray_from_alias(dataarray_alias)


# storing data array as dict for a dataset
class Datavars(StorageDict):
    '''
       @TypeSpec dict<<var_name:str>,values:models.DataArray>
    '''
    pass


def xr_dataarray_from_alias(dataarray_alias):
    da_ret = DataArray.get_by_alias(dataarray_alias)

    dims = da_ret.dims
    data = da_ret.values

    # use comprehension instead
    coords = dict()
    for k, v in da_ret.coords.items():
        coords[k] = np.asarray(da_ret.coords[k])

    return xr.DataArray(data, dims=dims, coords=coords)


def xr_dataarray_from_obj(dataarray_obj):
    da_ret = dataarray_obj
    dims = da_ret.dims
    data = da_ret.values
    coords = dict()
    for k, v in da_ret.coords.items():
        coords[k] = np.asarray(da_ret.coords[k])
    return xr.DataArray(data, dims=dims, coords=coords)


# dims here should be optional like xr.Dataset
class Dataset(StorageObj):
    '''
    @ClassField dims list<str>
    @ClassField coords models.Coordinates
    @ClassField data_vars models.Datavars
    '''

    @classmethod
    def from_xr_dataset(cls, xr_dataset):
        inst = cls()
        inst.dims = list(xr_dataset.dims)

        for coord_name, coord_val in xr_dataset.coords.items():
            if coord_name in xr_dataset.dims:  ## temp fix to assign arrays otherwise would need to
                # change coordinate model to have dataarray
                # or put it in datavar?
                inst.coords[coord_name] = coord_val.values

        for data_key, data_vals in xr_dataset.data_vars.items():
            inst.data_vars[data_key] = DataArray.from_xr_dataarray(data_vals)
        return inst

    @classmethod
    def from_file(cls, netcdf_or_ds_path):
        xr_dataset = xr.open_dataset(netcdf_or_ds_path)
        return cls.from_xr_dataset(xr_dataset)

    @classmethod
    def to_xr_dataset(cls, ds_alias):
        da_ret = cls.get_by_alias(ds_alias)

        additional_coords = dict()
        for k, v in da_ret.coords.items():
            additional_coords[k] = np.asarray(v)
        data_vars = {k: xr_dataarray_from_obj(da_ret.data_vars[k]) for k in da_ret.data_vars.keys()}
        return xr.Dataset(data_vars, coords=additional_coords)
