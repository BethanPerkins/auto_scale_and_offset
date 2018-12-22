import xarray as xr
import numpy as np

class ScaleOffsetCalculator:
    """
    Automatically calculate scale and offset for an xarray

    """

    def get(self, data, fill_value, dtype=np.uint16):
        """

        :param data:
        :param fill_value:
        :param dtype:
        :return:
        """

        # Mask out missing values
        fill_mask = data.values == fill_value

        # Get the max/min range of the final dataset. This is 90% of the range
        # available in the datatype
        dtype_range = np.iinfo(dtype).max - np.iinfo(dtype).min
        output_min = np.iinfo(dtype).min + dtype_range*0.05

        # Range of the passed data
        input_range = data[~fill_mask].max() - data[~fill_mask].min()
        input_min = data[~fill_mask].min()

        # Calculate offset required to bring the minimum data value to the
        # minimum output value
        offset = input_min - output_min

        # Inversley apply the offset
        intermediate_data = data - offset

        # Calculate the scale factor required to bring the data range to 90%
        # of the dtype range
        scale_factor = input_range / (dtype_range * 0.9)

        # Apply the scaling factor
        output_data = intermediate_data / scale_factor

        # Re-apply the fill value
        output_data[fill_mask] = fill_value

        return output_data, scale_factor, offset

