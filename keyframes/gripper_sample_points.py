import numpy as np
import pyquaternion as pyq


def qpos_to_T(quat, pos):
    rot = pyq.Quaternion(quat).rotation_matrix
    T = np.eye(4)
    T[:3,:3] = rot
    T[:3,3] = pos
    return T

class Base3dNode:

    def __init__(self, pos=np.array([0,0,0]), quat=np.array([1,0,0,0])):
        if type(pos) != np.ndarray:
            pos = np.array(pos)
        self.quat = quat
        self.pos = pos
        self.parent_T_node = qpos_to_T(quat, pos)
        self.world_T_node = None
        self.type = "base"

    def forward_T(self, world_T_parent):
        self.world_T_node = world_T_parent @ self.parent_T_node

class Body3dNode(Base3dNode):

    def __init__(self, pos=np.array([0,0,0]), quat=np.array([1,0,0,0])):
        super().__init__(pos, quat=quat)
        if type(pos) != np.ndarray:
            pos = np.array(pos)
        self.pos = pos
        self.children = []
        self.type = "body"

    def add_child(self, child):
        self.children.append(child)

    def forward_T(self, world_T_parent):
        super().forward_T(world_T_parent)
        for child in self.children:
            child.forward_T(self.world_T_node)

class Leaf3dNode(Base3dNode):

    def __init__(self, pos=np.array([0,0,0]), quat=np.array([1,0,0,0])):
        super().__init__(pos, quat=quat)
        self.type = "leaf"

    def create_surface_grid(self, cell_size: float):
        raise NotImplementedError

class BoxNode(Leaf3dNode):

    def __init__(self, sx_half, sy_half, sz_half, pos=np.array([0,0,0])):
        super().__init__(pos)
        self.sx = 2*sx_half
        self.sy = 2*sy_half
        self.sz = 2*sz_half
        self.v = self.create_box_vertices(self.sx, self.sy, self.sz)
        # NOTE faces are not triangles
        self.faces = self.create_box_faces()
        self.type = "box"

    def create_surface_grid(self, cell_size: float):
        grid = []
        for v in self.v:
            grid.append(v)
        for face in self.faces:
            v0, v1, v2, v3 = face
            p0, p1, p2, p3 = self.v[v0], self.v[v1], self.v[v2], self.v[v3]
            min_x = min(p0[0], p1[0], p2[0], p3[0])
            max_x = max(p0[0], p1[0], p2[0], p3[0])
            min_y = min(p0[1], p1[1], p2[1], p3[1])
            max_y = max(p0[1], p1[1], p2[1], p3[1])
            min_z = min(p0[2], p1[2], p2[2], p3[2])
            max_z = max(p0[2], p1[2], p2[2], p3[2])

            nx = max(int(np.ceil((max_x - min_x) / cell_size)), 1)
            ny = max(int(np.ceil((max_y - min_y) / cell_size)), 1)
            nz = max(int(np.ceil((max_z - min_z) / cell_size)), 1)
            
            for ix, x in enumerate(np.linspace(min_x, max_x, nx)):
                ix_is_extrema = ix == 0 or ix == nx - 1
                for iy, y in enumerate(np.linspace(min_y, max_y, ny)):
                    iy_is_extrema = iy == 0 or iy == ny - 1
                    for iz, z in enumerate(np.linspace(min_z, max_z, nz)):
                        iz_is_extrema = iz == 0 or iz == nz - 1
                        if ix_is_extrema and iy_is_extrema and iz_is_extrema:
                            continue
                        grid.append(np.array([x, y, z]))
        return np.vstack(grid)

    def create_box_vertices(self, sx, sy, sz):
        vertices = np.array([
            [-sx/2, -sy/2, -sz/2],  # vertex 0
            [sx/2, -sy/2, -sz/2],   # vertex 1
            [sx/2, sy/2, -sz/2],    # vertex 2
            [-sx/2, sy/2, -sz/2],   # vertex 3
            [-sx/2, -sy/2, sz/2],   # vertex 4
            [sx/2, -sy/2, sz/2],    # vertex 5
            [sx/2, sy/2, sz/2],     # vertex 6
            [-sx/2, sy/2, sz/2]     # vertex 7
        ])
        return vertices

    def create_box_faces(self):
        faces = np.array([
            [0, 1, 2, 3],  # face 0
            [1, 5, 6, 2],  # face 1
            [5, 4, 7, 6],  # face 2
            [4, 0, 3, 7],  # face 3
            [3, 2, 6, 7],  # face 4
            [0, 4, 5, 1]   # face 5
        ])
        return faces

class MeshNode(Leaf3dNode):

    def __init__(self, v, f, n, pos=np.array([0,0,0]), quat=np.array([1,0,0,0])):
        super().__init__(pos, quat=quat)
        self.v = v
        self.f = f
        self.n = n
        self.type = "mesh"

    def create_surface_grid(self, cell_size: float):
        return self.v
    


"""

<body name="right_hand" pos="0 0 0.0245" quat="0.707107 0 0 0.707107">
    <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
    <geom type="mesh" contype="1" conaffinity="1" group="1" rgba="0.5 0.1 0.1 1" pos= "0 0 0.03" mesh="eGripperBase" />

    <geom size="0.035 0.014" pos="0 0 0.015" type="cylinder" rgba="0 0 0 1"/>

    <body name="hand" pos="0 0 0.12" quat="-1 0 1 0">
        <camera name="behindGripper" mode="track" pos="0 0 -0.5" quat="0 1 0 0" fovy="60" />
        <camera name="gripperPOV" mode="track" pos="0 -0.1 0" quat="-1 -1.3 0 0" fovy="90" />

        <site name="endEffector" pos="0.04 0 0" size="0.01" rgba='1 1 1 0' />
        <geom name="rail" type="box" pos="-0.05 0 0" density="7850" size="0.005 0.055 0.005"  rgba="0.5 0.5 0.5 1.0" condim="3" friction="2 0.1 0.002"   />

        <!--IMPORTANT: For rougher contact with gripper, set higher friciton values for the other interacting objects -->
        <body name="rightclaw" pos="0 -0.05 0" >

            <geom class="base_col" name="rightclaw_it" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015"  rgba="1 1 1 1.0"   />

            <joint name="r_close" pos="0 0 0" axis="0 1 0" range= "0 0.04" armature="100" damping="1000" limited="true"  type="slide"/>
            <!-- <joint name="r_close" pos="0 0 0" axis="0 1 0" range= "0 0.03" armature="100" damping="1000" limited="true"  type="slide"/>  -->

            <!-- <site name="rightEndEffector" pos="0.0 0.005 0" size="0.044 0.008 0.012" type='box' /> -->

            <!-- <site name="rightEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
            <site name="rightEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/>
            <body name="rightpad" pos ="0 .003 0" >
                <geom name="rightpad_geom" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015" rgba="1 1 1 1.0" solimp="0.95 0.99 0.01" solref="0.01 1" friction="2 0.1 0.002" contype="1" conaffinity="1" mass="1"/>
            </body>

        </body>

        <body name="leftclaw" pos="0 0.05 0">
            <geom class="base_col" name="leftclaw_it" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015"  rgba="0 1 1 1.0"  />
            <joint name="l_close" pos="0 0 0" axis="0 1 0" range= "-0.03 0" armature="100" damping="1000" limited="true"  type="slide"/>
            <!-- <site name="leftEndEffector" pos="0.0 -0.005 0" size="0.044 0.008 0.012" type='box' /> -->
            <!-- <site name="leftEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
            <site name="leftEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/>
            <body name="leftpad" pos ="0 -.003 0" >
                <geom name="leftpad_geom" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015" rgba="0 1 1 1.0" solimp="0.95 0.99 0.01" solref="0.01 1" friction="2 0.1 0.002"  contype="1" conaffinity="1" />
            </body>

        </body>
    </body>
</body>

"""


def sample_box_vertices(root, cell_size: float):
    if root.type == "box" or root.type == "mesh":
        verts = root.create_surface_grid(cell_size)
        verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
        verts = root.world_T_node @ verts.T
        verts = verts.T
        return verts
    elif root.type == "body":
        verts = []
        for child in root.children:
            verts.extend(sample_box_vertices(child, cell_size))
        return np.vstack(verts)


class GripperSampler:

    def __init__(self, cell_size = 0.025) -> None:
        self.cell_size = cell_size
        self.num_sample_points = None


    def contruct_gripper(self, pad_dist = 0.088, offset = np.zeros(3)):
        """
        NOTE
        - z-pos of lower gripper area: 0.0
        - total height of gripper: 0.09 (pad height) + 0.054 (gripper_base height) - 0.005 (overlap) = 0.139
        - max gripper width: 0.1 (max outer distance between left and right pad) - 2 * 0.006 (pad thickness) = 0.088
        """
        gripper_total_height = 0.139
        pad_dist_max = 0.088
        assert pad_dist <= pad_dist_max, f"pad_dist must be less than {pad_dist_max}"

        right_hand_quat = pyq.Quaternion(axis=[1, 0, 0], angle=np.pi) # Rotate 180 about X
        # translate z by 0.026 such that z-pos of upper gripper is at 0.0
        right_hand = Body3dNode(np.array([0, 0, 0.026 + gripper_total_height]) + offset, quat=right_hand_quat.elements)

        # gripper base
        # z-pos of upper box area (gripper_base and pad overlap by 0.005): 0.027 + 0.053 = 0.08
        # z-pos of lower box area: 0.053 - 0.027 = 0.026
        gripper_base = BoxNode(0.033, 0.053, 0.027, [0, 0, 0.053])
        right_hand.add_child(gripper_base)

        # container of pads
        # z-pos of lower box area (note that gripper is rotated about X by 180 degrees): 0.12 - 0.045 = 0.075
        hand = Body3dNode(pos=[0,0,0.12], quat=[-1,0,1,0])  # [np.cos(np.pi/4),0,np.sin(np.pi/4),0])
        right_hand.add_child(hand)

        # rightpad
        y_offset = pad_dist / 2 + 0.003  # if pad_dist == pad_dist_max, then y_offset == 0.047 == 0.05 - 0.003
        rightpad = BoxNode(0.045, 0.003, 0.015, [0, -y_offset, 0])
        hand.add_child(rightpad)

        # leftpad
        leftpad = BoxNode(0.045, 0.003, 0.015, [0, y_offset, 0])
        hand.add_child(leftpad)

        return right_hand
    
    def sample(self, pad_dist = 0.088, offset = np.zeros(3)):
        right_hand = self.contruct_gripper(pad_dist=pad_dist, offset=offset)
        right_hand.forward_T(np.eye(4))
        surface_verts = sample_box_vertices(right_hand, self.cell_size)
        return surface_verts
    
    def get_num_sample_points(self):
        """
        Returns the number of sample points that will be generated by sample()
        """
        if self.num_sample_points is None:
            right_hand = self.contruct_gripper()
            right_hand.forward_T(np.eye(4))
            self.num_sample_points = sample_box_vertices(right_hand, self.cell_size).shape[0]
        return self.num_sample_points

    def load_gripper_mesh(self, path):
        # eGripperBase_path = "/home/user/Documents/projects/Metaworld/keyframes/test_data/eGripperBase.obj"
        # v_eGripperBase, _, n_eGripperBase, f_eGripperBase, _, _ = igl.read_obj(eGripperBase_path)
        # 
        pass