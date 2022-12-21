import bpy
import os
import numpy as np
import pickle
from mathutils import Vector, Quaternion
import json


SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15


def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues


def set_pose_from_rodrigues(armature, bone_name, rodrigues,
                            rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return


class SMPLXSetPoseshapes(bpy.types.Operator):
    bl_idname = "object.smplx_set_poseshapes"
    bl_label = "Update Pose Shapes"
    bl_description = ("Sets and updates corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    def rodrigues_to_mat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=object)
        return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Calculate weights of pose corrective blend shapes
    # Input is pose of all 55 joints, output is weights for all joints except pelvis
    def rodrigues_to_posecorrective_weight(self, pose):
        joints_posecorrective = NUM_SMPLX_JOINTS
        rod_rots = np.asarray(pose).reshape(joints_posecorrective, 3)
        mat_rots = [self.rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return(bshapes)

    def execute(self, context):
        obj = bpy.context.object

        # Get armature pose in rodrigues representation
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

        for index in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        poseweights = self.rodrigues_to_posecorrective_weight(pose)

        # Set weights for pose corrective shape keys
        for index, weight in enumerate(poseweights):
            obj.data.shape_keys.key_blocks["Pose%03d" % index].value = weight

        return {'FINISHED'}


class SMPLXUpdateJointLocations(bpy.types.Operator):
    bl_idname = "object.smplx_update_joint_locations"
    bl_label = "Update Joint Locations"
    bl_description = ("Update joint locations after shape/expression changes")
    bl_options = {'REGISTER', 'UNDO'}

    j_regressor_female = { 10: None, 300: None }
    j_regressor_male = { 10: None, 300: None }
    j_regressor_neutral = { 10: None, 300: None }

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def load_regressor(self, gender, betas):
        path = os.path.dirname(os.path.realpath(__file__))
        if betas == 10:
            suffix = ""
        elif betas == 300:
            suffix = "_300"
        else:
            print(f"ERROR: No betas-to-joints regressor for desired beta shapes [{betas}]")
            return (None, None)

        regressor_path = os.path.join(path, "data", f"smplx_betas_to_joints_{gender}{suffix}.json")
        with open(regressor_path) as f:
            data = json.load(f)
            return (np.asarray(data["betasJ_regr"]), np.asarray(data["template_J"]))

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        # Get beta shapes
        betas = []
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                betas.append(key_block.value)
        num_betas = len(betas)
        betas = np.array(betas)

        # Cache regressor files on first call
        for target_betas in [10, 300]:
            if self.j_regressor_female[target_betas] is None:
                self.j_regressor_female[target_betas] = self.load_regressor("female", target_betas)

            if self.j_regressor_male[target_betas] is None:
                self.j_regressor_male[target_betas] = self.load_regressor("male", target_betas)

            if self.j_regressor_neutral[target_betas] is None:
                self.j_regressor_neutral[target_betas] = self.load_regressor("neutral", target_betas)

        if "female" in obj.name:
            (betas_to_joints, template_j) = self.j_regressor_female[num_betas]
        elif "male" in obj.name:
            (betas_to_joints, template_j) = self.j_regressor_male[num_betas]
        else:
            (betas_to_joints, template_j) = self.j_regressor_neutral[num_betas]

        joint_locations = betas_to_joints @ betas + template_j

        # Set new bone joint locations
        armature = obj.parent
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        for index in range(NUM_SMPLX_JOINTS):
            bone = armature.data.edit_bones[SMPLX_JOINT_NAMES[index]]
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)

            # Convert SMPL-X joint locations to Blender joint locations
            joint_location_smplx = joint_locations[index]
            bone_start = Vector( (joint_location_smplx[0], -joint_location_smplx[2], joint_location_smplx[1]) )
            bone.translate(bone_start)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj

        return {'FINISHED'}


bpy.utils.register_class(SMPLXUpdateJointLocations)
bpy.utils.register_class(SMPLXSetPoseshapes)


def import_smplx(dir_path='./data/', pose_path='./example.pkl'):
    bpy.ops.wm.append(filename='SMPLX-mesh-neutral',
                      directory=dir_path + 'smplx_model_20210421.blend/Object')
    object_name = bpy.context.selected_objects[0].name
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
    bpy.data.objects[object_name].select_set(True)

    obj = bpy.context.object

    if obj.type == 'MESH':
        armature = obj.parent
    else:
        armature = obj
        obj = armature.children[0]
        bpy.context.view_layer.objects.active = obj

    path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(path, "data", "smplx_handposes.npz")
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
        hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

    print("Loading: " + pose_path)

    translation = None
    global_orient = None
    body_pose = None
    jaw_pose = None
    left_hand_pose = None
    right_hand_pose = None
    betas = None
    expression = None
    with open(pose_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

        if "transl" in data:
            translation = np.array(data["transl"]).reshape(3)

        if "global_orient" in data:
            global_orient = np.array(data["global_orient"]).reshape(3)

        body_pose = np.array(data["body_pose"])
        if body_pose.shape != (1, NUM_SMPLX_BODYJOINTS * 3):
            print(f"Invalid body pose dimensions: {body_pose.shape}")
            body_data = None
            return {'CANCELLED'}

        body_pose = np.array(data["body_pose"]).reshape(NUM_SMPLX_BODYJOINTS, 3)

        if "jaw_pose" in data:
            jaw_pose = np.array(data["jaw_pose"]).reshape(3)
        else:
            jaw_pose = np.zeros(3)
        left_hand_pose = np.array(data["left_hand_pose"]).reshape(-1, 3)
        right_hand_pose = np.array(data["right_hand_pose"]).reshape(-1, 3)

        betas = np.array(data["betas"]).reshape(-1).tolist()
        if "expression" in data:
            expression = np.array(data["expression"]).reshape(-1).tolist()
        else:
            expression = np.zeros(10).tolist()


    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"

        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    if global_orient is not None:
        set_pose_from_rodrigues(armature, "pelvis", global_orient)

    for index in range(NUM_SMPLX_BODYJOINTS):
        pose_rodrigues = body_pose[index]
        bone_name = SMPLX_JOINT_NAMES[index + 1] # body pose starts with left_hip
        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

    set_pose_from_rodrigues(armature, "jaw", jaw_pose)

    # # Left hand
    # start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3
    # for i in range(0, NUM_SMPLX_HANDJOINTS):
    #     pose_rodrigues = left_hand_pose[i]
    #     bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
    #     pose_relaxed_rodrigues = self.hand_pose_relaxed[i]
    #     set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

    # # Right hand
    # start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3 + NUM_SMPLX_HANDJOINTS
    # for i in range(0, NUM_SMPLX_HANDJOINTS):
    #     pose_rodrigues = right_hand_pose[i]
    #     bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
    #     pose_relaxed_rodrigues = self.hand_pose_relaxed[NUM_SMPLX_HANDJOINTS + i]
    #     set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

    # if translation is not None:
    #     # Set translation
    #     armature.location = (translation[0], -translation[2], translation[1])

    # Activate corrective poseshapes
    bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

    # Set face expression
    for index, exp in enumerate(expression):
        key_block_name = f"Exp{index:03}"

        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = exp
        else:
            print(f"ERROR: No key block for: {key_block_name}")

    return bpy.context.object.parent.name
