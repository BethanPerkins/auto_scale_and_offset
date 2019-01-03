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
        fill_mask = data == fill_value

        # Get the max/min range of the final dataset. This is 90% of the range
        # available in the datatype
        dtype_range = np.iinfo(dtype).max - np.iinfo(dtype).min
        target_max = np.iinfo(dtype).min + dtype_range * 0.95
        target_min = np.iinfo(dtype).min + dtype_range*0.05

        # Range of the passed data
        input_max = data[~fill_mask].max()
        input_min = data[~fill_mask].min()

        # Calculate scale factor
        scale_factor = (input_min - input_max) / (target_min - target_max)

        # Calculate offset
        offset = input_max - (target_max * scale_factor)

        # Cut both scale and offset down to 2 sig figures to avoid
        # over-specifying from small values
        scale_factor = float(('%.1e') % scale_factor)
        offset = float(('%.1e') % offset)

        # Inversly apply scale and offset
        output_data = (data - offset) / scale_factor

        # Conver to datatype
        output = output_data.astype(dtype)

        return output, scale_factor, offset


if __name__ == "__main__":

    soc = ScaleOffsetCalculator()

    soc.get(np.array([0, 100, 200]), 999)