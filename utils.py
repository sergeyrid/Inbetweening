import bpy
from smplx_utils import import_smplx
import retarget_addon


def retarget(sample_path='./sample.bvh', smplx_path='./data/',
             pose_path='./example.pkl', config_path='./smplx_to_bvh.rtconf'):
    smplx_object = bpy.data.objects[import_smplx(smplx_path, pose_path)]
    bpy.ops.import_anim.bvh(filepath=sample_path)
    sample_object = bpy.data.objects[bpy.context.object.name]
    s = retarget_addon.utilfuncs.state()
    s.selected_source = smplx_object
    s.update_source()
    bpy.ops.retarget.load(filepath=config_path)


def export_bvh(filepath):
    for _ in range(4):
        bpy.ops.export_anim.bvh(filepath=filepath, check_existing=False,
                                frame_start=1, frame_end=1, rotate_mode='ZYX',
                                root_transform_only=True)
        bpy.ops.import_anim.bvh(filepath=filepath)

retarget_addon.register()
