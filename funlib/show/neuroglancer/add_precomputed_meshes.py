import os

from neuroglancer.local_volume import InvalidObjectIdForMesh


def getHierarchicalMeshPath(object_id, hierarchical_size):

    assert object_id != 0

    level_dirs = []
    num_level = 0
    while object_id > 0:
        level_dirs.append(int(object_id % hierarchical_size))
        object_id = int(object_id / hierarchical_size)
    num_level = len(level_dirs) - 1
    level_dirs = [str(lv) for lv in reversed(level_dirs)]
    return os.path.join(str(num_level), *level_dirs)


def add_precomputed_meshes(
        layer,
        precomputed_mesh_path,
        precomputed_mesh_path_hierarchical_size,
        ):

    # if each mesh obj is ~10KB, 1M cache == 10GB of memory
    # @cached(cache=RRCache(maxsize=1024*1024))
    def get_object_mesh(object_id):

        if object_id == 0:
            raise InvalidObjectIdForMesh()

        mesh_path = precomputed_mesh_path + '/mesh'

        if precomputed_mesh_path_hierarchical_size:
            mesh_path = os.path.join(
                mesh_path,
                getHierarchicalMeshPath(
                    object_id, precomputed_mesh_path_hierarchical_size))
        else:
            mesh_path = os.path.join(mesh_path, str(object_id))

        try:
            file = open(mesh_path, mode='rb')
        except Exception as e:
            # Mesh file not generated
            raise InvalidObjectIdForMesh()

        mesh_obj = file.read()
        file.close()
        return mesh_obj

    layer.get_object_mesh = get_object_mesh
