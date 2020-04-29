from .scale_pyramid import ScalePyramid
import neuroglancer
from neuroglancer.local_volume import MeshesNotSupportedForVolume


def new_layer(
        array,
        units=None,
        axis_names=None,
        ):
    '''Create a layer which can be shared across contexts.

        array:

            A ``daisy``-like array, containing attributes ``roi``,
            ``voxel_size``, and ``data``. If a list of arrays is given, a
            ``ScalePyramid`` layer is generated.

        axis_names:

            Names of the axes in the data (e.g. ['t', 'z', 'y', 'x'])
            Defaults to the last n elements in ['t', 'c', 'z', 'y', 'x']
            where n = number of dimensions in the array.

        units:

            List of strings representing the units of each axis. Defaults
            to 'nm' for spatial dimensions and '' for other dimensions.
    '''

    is_multiscale = type(array) == list

    if is_multiscale:
        ndim = len(array[0].shape)
    else:
        ndim = len(array.shape)

    if not axis_names:
        axis_names = ['t^', 'c^', 'z', 'y', 'x'][-1*ndim:]

    if not units:
        # default of nm
        units = ['', '', 'nm', 'nm', 'nm'][-1*ndim:]

    if is_multiscale:

        local_volumes = []

        for v in array:

            scales = v.voxel_size
            while len(scales) < len(axis_names):
                scales = (1,) + scales

            voxel_offset = v.roi.get_offset() / v.voxel_size
            while len(voxel_offset) < len(axis_names):
                voxel_offset = (0,) + voxel_offset

            local_vol = neuroglancer.LocalVolume(
                data=v.data,
                dimensions=neuroglancer.CoordinateSpace(
                    scales=scales,
                    units=units,
                    names=axis_names),
                )
            setattr(local_vol, "voxel_size", v.voxel_size)
            setattr(local_vol, "pyramid_voxel_offset", voxel_offset)
            local_volumes.append(local_vol)

        layer = ScalePyramid(
            local_volumes,
            )

    else:

        scales = array.voxel_size
        while len(scales) < len(axis_names):
            scales = (1,) + scales

        voxel_offset = array.roi.get_offset() / array.voxel_size
        while len(voxel_offset) < len(axis_names):
            voxel_offset = (0,) + voxel_offset

        layer = neuroglancer.LocalVolume(
            data=array.data,
            dimensions=neuroglancer.CoordinateSpace(
                scales=scales,
                units=units,
                names=axis_names),
            voxel_offset=voxel_offset)
        setattr(layer, "voxel_size", array.voxel_size)

    return layer


def add_layer(
        context,
        array,
        name,
        layer=None,
        opacity=None,
        shader=None,
        visible=True,
        scale_rgb=False,
        c=[0, 1, 2],
        h=[0.0, 0.0, 1.0],
        axis_names=None,
        units=None,
        ondemand_mesh_scale_key=(1, 1, 1),
        ):

    '''Add a layer to a neuroglancer context.

    Args:

        context:

            The neuroglancer context to add a layer to, as obtained by
            ``viewer.txn()``.

        array:

            A ``daisy``-like array, containing attributes ``roi``,
            ``voxel_size``, and ``data``. If a list of arrays is given, a
            ``ScalePyramid`` layer is generated.

        name:

            The name of the layer.

        opacity:

            A float to define the layer opacity between 0 and 1

        shader:

            A string to be used as the shader. If set to ``'rgb'``, an RGB
            shader will be used. Other options include 'rgba', 'heatmap',
            and 'mask'.

        visible:

            A bool which defines layer visibility

        scale_rgb:

            Multiply the RGB vector by 255

        c (channel):

            A list of ints to define which channels to use for an rgb shader

        h (hue):

            A list of floats to define rgb color for an rgba shader

        axis_names:

            Names of the axes in the data (e.g. ['t', 'z', 'y', 'x'])
            Defaults to the last n elements in ['t', 'c', 'z', 'y', 'x']
            where n = number of dimensions in the array.

        units:

            List of strings representing the units of each axis. Defaults
            to 'nm' for spatial dimensions and '' for other dimensions.

        ondemand_mesh_scale_key:

            Tuple of the scale key to be used for neuroglancer's built-in
            on-demand meshing engine.
            Default: (1, 1, 1), using the highest resolution data available.
            Set to `None` to disable meshing entirely (useful for very big
            dataset to avoid accidental meshing).
    '''

    if context.dimensions.rank == 0:
        # add default viewing dimension
        context.dimensions = neuroglancer.CoordinateSpace(
            names=['x', 'y', 'z'],
            units='nm',
            scales=[4, 4, 40])

    is_multiscale = type(array) == list

    if shader is None:
        a = array if not is_multiscale else array[0]
        dims = a.roi.dims()
        if dims < len(a.data.shape):
            channels = a.data.shape[0]
            if channels > 1:
                shader = 'rgb'

    if shader == 'rgb':
        if scale_rgb:
            shader = """
void main() {
    emitRGB(
        255.0*vec3(
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)))
        );
}""" % (c[0], c[1], c[2])

        else:
            shader = """
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)))
        );
}""" % (c[0], c[1], c[2])

    elif shader == 'rgba':
        shader = """
void main() {
    emitRGBA(
        vec4(
        %f, %f, %f,
        toNormalized(getDataValue()))
        );
}""" % (h[0], h[1], h[2])

    elif shader == 'mask':
        shader = """
void main() {
  emitGrayscale(255.0*toNormalized(getDataValue()));
}"""

    elif shader == 'heatmap':
        shader = """
void main() {
    float v = toNormalized(getDataValue(0));
    vec4 rgba = vec4(0,0,0,0);
    if (v != 0.0) {
        rgba = vec4(colormapJet(v), 1.0);
    }
    emitRGBA(rgba);
}"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader
    if opacity is not None:
        kwargs['opacity'] = opacity

    if layer is None:
        layer = new_layer(
            array,
            units=units,
            axis_names=axis_names,
            )

    if ondemand_mesh_scale_key is None:
        # disabling meshing
        def get_object_mesh(object_id):
            raise MeshesNotSupportedForVolume()
        layer.get_object_mesh = get_object_mesh
    else:
        if is_multiscale:
            layer.set_ondemand_mesh_scale_key(ondemand_mesh_scale_key)

    context.layers.append(
            name=name,
            layer=layer,
            visible=visible,
            **kwargs)
