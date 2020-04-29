import neuroglancer
import operator
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_MAX_DOWNSAMPLING = 64
DEFAULT_MAX_DOWNSAMPLED_SIZE = 128
DEFAULT_MAX_DOWNSAMPLING_SCALES = float('inf')


def compute_near_isotropic_downsampling_scales(
        size,
        voxel_size,
        dimensions_to_downsample,
        max_scales=DEFAULT_MAX_DOWNSAMPLING_SCALES,
        max_downsampling=DEFAULT_MAX_DOWNSAMPLING,
        max_downsampled_size=DEFAULT_MAX_DOWNSAMPLED_SIZE):
    """Compute a list of successive downsampling factors.
    https://github.com/google/neuroglancer/blob/c9a6b9948dd416997c91e655ec3d67bf6b7e771b/python/neuroglancer/downsample_scales.py
    """

    num_dims = len(voxel_size)
    cur_scale = np.ones((num_dims, ), dtype=int)
    scales = [tuple(cur_scale)]
    while (len(scales) < max_scales and
           (np.prod(cur_scale) < max_downsampling) and
           (size / cur_scale).max() > max_downsampled_size):
        # Find dimension with smallest voxelsize.
        cur_voxel_size = cur_scale * voxel_size
        smallest_cur_voxel_size_dim = dimensions_to_downsample[
            np.argmin(cur_voxel_size[dimensions_to_downsample])]
        cur_scale[smallest_cur_voxel_size_dim] *= 2
        target_voxel_size = cur_voxel_size[smallest_cur_voxel_size_dim] * 2
        for d in dimensions_to_downsample:
            if d == smallest_cur_voxel_size_dim:
                continue
            d_voxel_size = cur_voxel_size[d]
            if (abs(d_voxel_size - target_voxel_size) >
                    abs(d_voxel_size * 2 - target_voxel_size)):
                cur_scale[d] *= 2
        scales.append(tuple(cur_scale))
    return scales


class ScalePyramid(neuroglancer.LocalVolume):
    '''A neuroglancer layer that provides volume data on different scales.
    Mimics a LocalVolume.

    Args:

            volume_layers (``list`` of ``LocalVolume``):

                One ``LocalVolume`` per provided resolution.
    '''

    def __init__(
            self,
            volume_layers,
            ):

        super(neuroglancer.LocalVolume, self).__init__()

        logger.debug("Creating scale pyramid...")

        voxel_sizes = []
        for l in volume_layers:
            voxel_sizes.append([int(k) for k in l.voxel_size])

        min_voxel_size = min(voxel_sizes)

        self.volume_layers = {
            tuple(int(k) for k in tuple(map(operator.truediv,
                                            l.voxel_size,
                                            min_voxel_size))): l
            for l in volume_layers
        }

        self.ref_layer_key = (1,) * len(min_voxel_size)
        self.ref_layer_key_str = ','.join(
                                    [str(i) for i in (self.ref_layer_key)])
        self.ref_layer = self.volume_layers[self.ref_layer_key]

        self.total_dim_length = len(self.ref_layer.voxel_offset)

        assert len(self.ref_layer.voxel_size) == 3  # TODO: support other dims
        self.ref_voxel_size = self.ref_layer.voxel_size

        self.spatial_index_slice = slice(
            self.total_dim_length-3,
            self.total_dim_length)

        self.scale_pyramid_map = \
            self.compute_near_isotropic_downsampling_scales()

        self.ondemand_mesh_scale_key = self.ref_layer_key

        logger.debug(self.info())
        logger.debug("min_voxel_size: %s", min_voxel_size)
        logger.debug("provided scale keys: %s", self.volume_layers.keys())
        logger.debug("Pyramid map: %s" % self.scale_pyramid_map)

    def set_ondemand_mesh_scale_key(self, key):
        if key not in self.volume_layers:
            print("Available mesh scales:", self.volume_layers.keys())
            raise RuntimeError("Unavailable mesh key: %s" % str(key))
        self.ondemand_mesh_scale_key = key

    @property
    def volume_type(self):
        return self.volume_layers[self.ref_layer_key].volume_type

    @property
    def token(self):
        return self.volume_layers[self.ref_layer_key].token

    def info(self):

        reference_layer = self.volume_layers[self.ref_layer_key]

        info = {
            'dataType': reference_layer.data_type,
            'encoding': reference_layer.encoding,
            'generation': reference_layer.change_count,
            'coordinateSpace': reference_layer.dimensions.to_json(),
            'shape': reference_layer.shape,
            'volumeType': reference_layer.volume_type,
            'voxelOffset': reference_layer.voxel_offset,
            'chunkLayout': reference_layer.chunk_layout,
            'downsamplingLayout': reference_layer.downsampling_layout,
            'maxDownsampling': None,  # product of all dims is not limited
            'maxDownsampledSize': (
                None
                if math.isinf(reference_layer.max_downsampled_size)
                else reference_layer.max_downsampled_size),
            'maxDownsamplingScales': len(self.scale_pyramid_map),
            'maxVoxelsPerChunkLog2': 18
        }

        return info

    def get_encoded_subvolume(
            self, data_format, start, end, scale_key=None):

        if scale_key is None:
            scale_key = self.ref_layer_key_str

        assert scale_key in self.scale_pyramid_map
        scale, ds_factor, voxel_offset = self.scale_pyramid_map[scale_key]

        # adjust ROI which is calculated from ref layer to selected ds layer
        start -= voxel_offset
        end -= voxel_offset

        return self.volume_layers[scale].get_encoded_subvolume(
            data_format,
            start,
            end,
            scale_key=ds_factor,
            )

    def get_object_mesh(self, obj_id):
        return self.volume_layers[
            self.ondemand_mesh_scale_key].get_object_mesh(obj_id)

    def invalidate(self):
        return self.volume_layers[self.ref_layer_key].invalidate()

    def compute_near_isotropic_downsampling_scales(
            self,
            max_repeated=1,
            max_scales=9):
        '''Construct a pyramid scale conforming to what NG expects.
        Since the user provided layers may not match exactly to expectations,
        we will match the expected scale to the closest provided layers.

        A heuristic is used to stop generating higher downsampling scales.
        This is controlled by the max_repeated argument
        '''

        pyramid_map = dict()

        expected_scales = compute_near_isotropic_downsampling_scales(
            size=100000000000,
            voxel_size=self.ref_voxel_size,
            dimensions_to_downsample=[0, 1, 2],  # TODO: support non-3D volumes
            max_downsampling=float('inf'),
            max_scales=max_scales
            )

        last_closest = None
        repeat_counter = 0

        for scale in expected_scales:

            closest_scale = self.get_closest_available_scale(scale)
            if closest_scale == last_closest:
                repeat_counter += 1
                if repeat_counter > max_repeated:
                    break
            else:
                repeat_counter = 0
            last_closest = closest_scale

            ds_factor = (
                np.array(scale)/np.array(closest_scale)).astype(int).tolist()

            voxel_offset = self.get_voxel_offset_to_abs(
                self.volume_layers[closest_scale],
                ds_factor)

            # adjust parameters to include non-dim scales that NG expects
            while len(voxel_offset) < self.total_dim_length:
                voxel_offset.insert(0, 0)
            while len(ds_factor) < self.total_dim_length:
                ds_factor.insert(0, 1)
            ds_factor = ','.join([str(k) for k in ds_factor])

            while len(scale) < self.total_dim_length:
                scale = (1,) + scale
            scale = ','.join([str(i) for i in scale])

            pyramid_map[scale] = (closest_scale, ds_factor, voxel_offset)

        return pyramid_map

    def get_closest_available_scale(self, target_scale):

        best_prod = 0
        best_scale = None
        for scale in self.volume_layers.keys():
            ds_ratios = np.array(scale)/np.array(target_scale)
            if np.any(ds_ratios > 1):
                continue
            prod = np.prod(scale)
            if prod > best_prod:
                best_prod = prod
                best_scale = scale

        assert best_scale is not None
        return best_scale

    def get_voxel_offset_to_abs(self, layer, ds_factor):

        voxel_offset = layer.pyramid_voxel_offset[self.spatial_index_slice]
        assert len(ds_factor) == len(voxel_offset)
        voxel_offset = [
            int(n/m) for n, m in zip(voxel_offset, ds_factor)]
        return [int(k) for k in voxel_offset]
