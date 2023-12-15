import pythreejs as py3
import pyquaternion as pq
import numpy as np
from IPython.display import display
from ipywidgets import HTML, Output, VBox, jslink
import matplotlib.pyplot as plt
import matplotlib as mpl
import igl

from dataclasses import dataclass

import ipyevents as ipyev

from . import utils_viewer


# @dataclass
class Control_State:
    # keys_pressed : set = set()
    # special_keys_pressed : set = set()
    # mousedown : bool = False
    # key_callback_pairs : dict = {}
    # mouse_callbacks = []
    # last_key : str = None

    special_keys_key1 = ["alt", "ctrl", "meta", "shift"]
    special_keys_key2 = ["alt", "control", "meta", "shift"]

    def __init__(self, viewer, out):
        self.viewer = viewer
        self.out = out

        self.keys_pressed: set = set()
        self.special_keys_pressed: set = set()
        self.mousedown: bool = False
        self.key_callback_pairs: dict = {}
        self.mouse_callbacks = []
        self.last_key : str = None

    def handle_event(self, event):
        with self.out:
            # print(event.items())
            # print(event["event"])
            if event["event"] == "mousemove":
                pass
            elif event["event"] == "mousedown":
                self.mousedown = True
                # TODO If "Soft Vertex Transform" is selected and mouse is over vertex:
                # - Disable orbit control
                # - Move vertex
            elif event["event"] == "mouseup":
                self.mousedown = False

            for mc in self.mouse_callbacks:
                mc(event, self.mousedown, self)

            if event["event"] == "keydown":
                # if not event["repeat"]:
                #     if len(event["key"]) == 1 and event["key"].is:
                #         self.keys_pressed.add(event["key"])
                #     else:
                self.keys_pressed.clear()
                if event["key"].lower() not in self.special_keys_key2:
                    self.keys_pressed.add(event["key"].lower())


            elif event["event"] == "keyup":
                self.keys_pressed.clear()

            self.special_keys_pressed.clear()
            for spkey in self.special_keys_key1:
                if event[spkey + "Key"]:
                    self.special_keys_pressed.add(spkey.lower())
                    # self.special_keys_pressed.add(event["key"])

            send_on_repeat = False
            if "alt" in self.special_keys_pressed and "ctrl" in self.special_keys_pressed and "shift" in self.special_keys_pressed:
                send_on_repeat = True

            # generate keys

            gen_key = self.generate_key()

            # # If last keys is not in keys anymore => callback
            if self.last_key == gen_key and send_on_repeat:
                if gen_key in self.key_callback_pairs:
                    self.key_callback_pairs[gen_key](True, self)

            if self.last_key != gen_key:
                # print("genkey:" ,gen_key)
                if self.last_key in self.key_callback_pairs:
                    self.key_callback_pairs[self.last_key](False, self)

                if gen_key in self.key_callback_pairs:
                    self.key_callback_pairs[gen_key](True, self)

            self.last_key = gen_key

    def generate_key(self):
        key_join = '+'.join(np.concatenate([np.sort(list(self.special_keys_pressed)),
                                            np.sort(list(self.keys_pressed))]))
        return key_join

    def add_callback(self, key, fun):
        """
        key has form: key = [mouse=[mousedown[+...]],][special=[Shift[+...],]
        """
        self.key_callback_pairs[key] = fun


class Viewer:

    def __init__(self, view_width=800, view_height=800, config=None, out=None):
        self.view_width = view_width
        self.view_height = view_height
        self.config = config
        self.objects = {}
        self.num_added_objects = 0

        self.camera = py3.PerspectiveCamera(
            position=[0, 0, 5], 
            up=[0, 1, 0], 
            aspect=self.view_width / self.view_height,
            children=[py3.DirectionalLight(color='white', position=[3, 5, 1], intensity=0.5)]
        )
        self.camera_control = py3.OrbitControls(controlling=self.camera)
        self.camera_control.enableRotate = False
        self.scene = py3.Scene(children=[self.camera, py3.AmbientLight(color='#777777')])
        self.renderer = py3.Renderer(camera=self.camera,
                                     scene=self.scene,
                                     width=self.view_width, height=self.view_height,
                                     controls=[self.camera_control],
                                     antialias=True,
                                     alpha=True)

        # Event handler
        self.info = HTML('Event info')
        # self.event = ipyev.Event(source=self.renderer,
        #                          watched_events=['click', 'keydown', 'keyup', 'mousemove', 'mouseenter', 'mousedown',
        #                                          'mouseup'])  # , 'mousemove', 'mouseenter', 'mousedown', 'mouseup'
        # self.event.on_dom_event(self.handle_event)

        self.control_state = Control_State(self, out)

        # self.control_state.add_callback("ctrl+shift+o", self.update_camera_control_enabled)
        # self.control_state.add_callback("alt", self.toggle_camera_control)

        self.out = out

        # self.plane_transform = py3.Plane(normal=[1,1,0.2], constant=3)
        # self.plane_transform_helper = py3.PlaneHelper(self.plane_transform, 1, "red" )

    def update_camera_control_enabled(self, is_active, control_state):
        with self.out:
            print("update_camera_control_enabled", is_active)
        self.camera_control.enabled = not is_active

    def toggle_camera_control(self, toggle_val, control_state):
        with self.out:
            # self.camera_control.enabled = not self.camera_control.enabled
            # self.camera_control.noRotate = not self.camera_control.noRotate
            self.camera_control.enableRotate = toggle_val
            # self.camera_control.enablePan = not self.camera_control.enablePan
            # self.camera_control.enableKeys = not self.camera_control.enableKeys
            # self.camera_control.enableZoom = not self.camera_control.enableZoom

    def handle_event(self, event):
        lines = ['{}: {}'.format(k, v) for k, v in event.items()]
        content = '<br>'.join(lines)
        with self.out:
            self.control_state.handle_event(event)

        # for k,o in self.objects.items():
        #     # o.handle_event(event, self.keys, self.info)
        #     with self.out:
        #         o.handle_event(event, self.keys, self.info)
        #         print("object:",k)

        # self.info.value = content

    def remove(self, name):
        for o in self.objects[name].objects:
            self.scene.remove(o)

        # for c in self.renderer.controls:
        #     self.renderer.controls.remove(c)
        # self.renderer.controls = self.renderer.controls

    def add_obj(self, obj, name=None, mouse_callback=None):
        """
        How to update things?
        buffer_geom.attributes["position"].array = V
        """
        if name is None:
            name = self.num_added_objects

        # NOTE buffer_geom = buffer_mesh.geometry
        self.objects[name] = obj

        for o in obj.objects:
            self.scene.add(o)

        for k in obj.keymaps:
            self.control_state.add_callback(k[0], k[1])

        # TODO make with key
        if mouse_callback is not None:
            self.control_state.mouse_callbacks.append(mouse_callback)

        self.num_added_objects += 1

        # NOTE += does not work here => TODO why?
        self.renderer.controls = self.renderer.controls + obj.controls

        return name


class Scatter:
    # is_selectable : bool
    # is_selected : np.ndarray
    # selected_single_id : np.int32 = -1
    #
    # points : list
    # num_points : int
    # size : float
    # radius_select : float
    radius_scale: float = 1.0
    radius_scale_count = 0

    # is_add_mode : bool = False
    # is_remove_mode : bool = False
    #
    # is_draw_mode : bool = False
    # is_draw_mode_single : bool = False

    color_default = (0.37647059, 0.49019608, 0.54509804)
    color_multi = (0, 1, 0)
    color_single = (0, 0, 1)
    color_highlight = (1., 0.34117647, 0.13333333)
    # color_highlight = (0.        , 0.7372549 , 0.83137255)
    color_selected = (1., 0.92156863, 0.23137255)

    # on_select_hooks : list = []
    # on_select_hooks_single : list = []

    def __init__(self, V, out, size=0.1, radius_select=0.1, is_selectable=True):
        self.out = out
        self.is_selectable = is_selectable
        self.num_points = V.shape[0]
        self.size = size
        self.radius_select = radius_select

        self.is_selectable: bool
        self.is_selected: np.ndarray
        self.selected_single_id: np.int32 = -1

        self.points: list = []
        self.is_add_mode: bool = False
        self.is_remove_mode: bool = False

        self.is_draw_mode: bool = False
        self.is_draw_mode_single: bool = False

        buffer_geom = py3.BufferGeometry(attributes=dict(
            position=py3.BufferAttribute(V.astype(np.float32), normalized=False),
            color=py3.BufferAttribute(
                np.repeat(np.expand_dims(np.array(self.color_default, dtype=np.float32), axis=0), self.num_points,
                          axis=0), normalized=False),
        ))
        material = py3.PointsMaterial(vertexColors='VertexColors', size=size)

        self.point_group = py3.Points(geometry=buffer_geom, material=material)
        self.point_group.geometry.exec_three_obj_method("computeBoundingSphere")
        self.is_selected = np.full(self.num_points, False)

        self.pencil = py3.Mesh(geometry=py3.SphereGeometry(radius=self.radius_select),
                               material=py3.MeshLambertMaterial(color='hotpink'))
        self.pencil.visible = False

        # self.point_group = py3.Group()
        # for i in range(self.num_points):
        #     point = py3.Mesh(geometry=py3.SphereGeometry(radius=self.size),
        #                     material=py3.MeshLambertMaterial(color='red'))
        #     point.position = tuple(V[i])
        #     point.is_selected = False
        #     point.point_id = i
        #     self.point_group.add(point)

        # self.drawer = py3.Picker(controlling=self.point_group, event='mousemove')
        # self.drawer.observe(self.on_select, names="point")

        self.objects = [self.point_group, self.pencil]
        self.controls = []
        self.keymaps = [("ctrl+d", self.toggle_select),
                        ("ctrl+r", self.toggle_remove_mode),
                        ("alt+s", self.toggle_visibility),
                        ("ctrl+shift+arrowup", self.radius_increase),
                        ("ctrl+shift+arrowdown", self.radius_decrease),
                        ("ctrl+shift+d", self.toggle_select_single)]

    def on_select(self, change):
        with self.out:
            owner = change["owner"]

            if owner.object is not None:
                point = np.array(self.pencil.position)
                pts_in_radius, norms = self.get_pts_in_radius(point,
                                                              self.point_group.geometry.attributes["position"].array)

                if self.is_draw_mode:
                    if not self.pencil.visible:
                        self.pencil.visible = True
                    self.is_selected[pts_in_radius] = not self.is_remove_mode
                    for hook in self.on_select_hooks:
                        hook(self.is_selected)
                    self.update_color()
                elif self.is_draw_mode_single:
                    if not self.pencil.visible:
                        self.pencil.visible = True
                    vid = np.argmin(norms)
                    if pts_in_radius[vid]:
                        if vid != self.selected_single_id:
                            self.selected_single_id = vid
                            for hook in self.on_select_hooks_single:
                                hook(self.selected_single_id)
                            self.update_color()
                        else:
                            self.selected_single_id = -1
            else:
                self.selected_single_id = -1
                self.pencil.visible = False

    def update_color(self):
        colors = self.point_group.geometry.attributes["color"].array
        colors[:, :] = self.color_default
        if self.is_draw_mode:
            colors[self.is_selected] = self.color_multi
        elif self.is_draw_mode_single:
            colors[self.selected_single_id] = self.color_single
        self.point_group.geometry.attributes["color"].array = colors
        self.point_group.geometry.attributes["color"].needsUpdate = True

    def get_pts_in_radius(self, point, points):
        # TODO should be searched with raycaster
        norms = np.linalg.norm(points - point, axis=-1)
        points_in_radius = norms < self.radius_select * self.radius_scale
        return points_in_radius, norms

    def mouse_over_vertex(self, point, tol):
        pts_in_radius, norms = self.get_pts_in_radius(point, self.point_group.geometry.attributes["position"].array)
        vid = np.argmin(norms)
        if norms[vid] < tol:
            return vid
        else:
            return -1

    def highlight_vertex(self, vid, color_highlight):
        colors = self.point_group.geometry.attributes["color"].array
        colors[:, :] = self.color_default
        colors[vid, :] = color_highlight
        self.point_group.geometry.attributes["color"].array = colors
        self.point_group.geometry.attributes["color"].needsUpdate = True

    def toggle_select(self, toggle_val, control_state):
        if toggle_val:
            self.is_draw_mode = not self.is_draw_mode
            self.pencil.visible = self.is_draw_mode
            if self.is_draw_mode:
                self.is_draw_mode_single = False
                self.update_color()

    def toggle_select_single(self, toggle_val, control_state):
        if toggle_val:
            self.is_draw_mode_single = not self.is_draw_mode_single
            self.pencil.visible = self.is_draw_mode_single
            if self.is_draw_mode_single:
                self.is_draw_mode = False
                self.update_color()

    def toggle_remove_mode(self, toggle_val, control_state):
        if toggle_val:
            self.is_remove_mode = not self.is_remove_mode

    def toggle_visibility(self, toggle_val, control_state):
        if toggle_val:
            self.point_group.visible = not self.point_group.visible

    def radius_increase(self, toggle_val, control_state):
        if toggle_val:
            self.radius_scale_count += 1
            self.radius_scale = np.exp(0.1 * self.radius_scale_count)
            self.pencil.scale = (self.radius_scale, self.radius_scale, self.radius_scale)
            # self.pencil.geometry.radius = self.radius_select

    def radius_decrease(self, toggle_val, control_state):
        with self.out:
            if toggle_val:
                self.radius_scale_count -= 1
                self.radius_scale = np.exp(0.1 * self.radius_scale_count)
                self.pencil.scale = (self.radius_scale, self.radius_scale, self.radius_scale)
                # self.pencil.geometry.radius.needsUpdate = True

    def update(self, V=None, is_selected=None):
        if V is not None:
            self.point_group.geometry.attributes["position"].array = V.astype(np.float32)
            self.point_group.geometry.attributes["position"].needsUpdate = True
            self.point_group.geometry.exec_three_obj_method("computeBoundingSphere")

        if is_selected is not None:
            self.is_selected = is_selected.copy()
            colors = self.point_group.geometry.attributes["color"].array
            colors[is_selected] = (0, 1, 0)
            colors[np.logical_not(is_selected)] = (1, 0, 0)
            self.point_group.geometry.attributes["color"].array = colors
            self.point_group.geometry.attributes["color"].needsUpdate = True
            for hook in self.on_select_hooks:
                hook(self.is_selected)

    def update2(self, V=None, color=None):
        with self.out:
            if V is not None:
                self.point_group.geometry.attributes["position"].array = V.astype(np.float32)
                self.point_group.geometry.attributes["position"].needsUpdate = True
                self.point_group.geometry.exec_three_obj_method("computeBoundingSphere")
            if color is not None:
                self.point_group.geometry.attributes["color"].array = color
                self.point_group.geometry.attributes["color"].needsUpdate = True


class BasicMesh:

    def __init__(self, V, T, out=None, color="white", flat_shading=False, opacity=1.0) -> None:
        self.flat_shading = flat_shading
        self.out = out

        with self.out:

            self.num_vertices = V.shape[0]
            self.num_faces = T.shape[0]
            self.T = T
            self.flat_shading = flat_shading

            self.opacity = opacity

            self.coloring = self.get_coloring(color)
            color = self.get_color(color, self.coloring)
            vertices = self.get_vertices(V, self.coloring)
            faces = self.get_faces(T, self.coloring)

            # geometry & material
            print("creating buffer and material...")
            if self.coloring == "uniform":
                buffer_geom = py3.BufferGeometry(attributes=dict(
                    position=py3.BufferAttribute(vertices, normalized=False),
                    index=py3.BufferAttribute(faces, normalized=False),
                ))
                material = py3.MeshPhongMaterial(color=color, side="DoubleSide", flatShading=flat_shading,
                                                 opacity=opacity, transparent=True)
            elif self.coloring in ["face", "vertex", "per_edge_per_face"]:
                buffer_geom = py3.BufferGeometry(attributes=dict(
                    position=py3.BufferAttribute(vertices, normalized=False),
                    index=py3.BufferAttribute(faces, normalized=False),
                    color=py3.BufferAttribute(color, normalized=False),
                ))
                material = py3.MeshPhongMaterial(vertexColors='VertexColors', side="DoubleSide",
                                                 flatShading=flat_shading, opacity=opacity, transparent=True)

            if not self.flat_shading:
                buffer_geom.exec_three_obj_method('computeVertexNormals')

            print("creating mesh...")
            self.mesh = py3.Mesh(
                geometry=buffer_geom,
                material=material
            )

            self.objects = [self.mesh]
            self.controls = []
            self.keymaps = []
            # self.objects = [self.mesh, self.hover_point]
            # self.controls = [self.drawer]
            # self.keymaps = [] + self.scatter.keymaps

    def update(self, V=None, color=None):
        if V is not None:
            vertices = self.get_vertices(V, self.coloring)
            self.update_vertices(vertices)
        if color is not None:
            with self.out:
                color = self.get_color(color, self.coloring)
                self.update_color(color, self.coloring)

    def update_faces(self, faces):
        self.mesh.geometry.attributes["index"].array = faces
        self.mesh.geometry.attributes["index"].needsUpdate = True

    def update_vertices(self, vertices):
        self.mesh.geometry.attributes["position"].array = vertices
        self.mesh.geometry.attributes["position"].needsUpdate = True
        self.mesh.geometry.exec_three_obj_method("computeBoundingSphere")

    def update_color(self, color, coloring):
        if coloring == "uniform":
            self.mesh.material.color = color
        elif coloring in ["vertex", "face", "per_edge_per_face"]:
            self.mesh.geometry.attributes["color"].array = color
            self.mesh.geometry.attributes["color"].needsUpdate = True

    def get_coloring(self, color):
        if type(color) == str:
            coloring = "uniform"
        elif type(color) == np.ndarray:
            if len(color.shape) == 2 and color.shape[0] == self.num_faces:
                coloring = "face"
            elif color.shape[0] == self.num_vertices:
                coloring = "vertex"
            elif len(color.shape) == 3 and color.shape[0] == self.num_faces:
                coloring = "per_edge_per_face"
            else:
                coloring = False
        else:
            coloring = False
        return coloring

    def get_faces(self, T, coloring):
        if coloring == "face":
            faces = np.linspace(0, T.shape[0] * 3 - 1, T.shape[0] * 3, dtype=np.uint16)
        elif coloring == "per_edge_per_face":
            faces = np.linspace(0, T.shape[0] * 3 * 3 - 1, T.shape[0] * 3 * 3, dtype=np.uint16)
        elif coloring in ["uniform", "vertex"]:
            faces = T.flatten().astype(np.uint16)
        return faces

    def get_vertices(self, V, coloring):
        if coloring == "face":
            vertices = V[self.T.flatten(), :].astype(np.float32)
        elif coloring == "per_edge_per_face":
            V_epi = np.sum(V[self.T, :], axis=1) / 3
            vertices = np.ndarray((self.T.shape[0] * 3 * 3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(2):
                    v = (i + j) % 3
                    vertices[i * 3 + j::9, :] = V[self.T[:, v], :]
                j = 2
                vertices[i * 3 + j::9, :] = V_epi
            vertices = vertices.astype(np.float32)
        elif coloring in ["uniform", "vertex"]:
            vertices = V.astype(np.float32)
        return vertices

    def get_color(self, color, coloring):
        if coloring == "face":
            color = np.repeat(color, 3, axis=0).astype(np.float32)
        elif coloring == "per_edge_per_face":
            color = color.reshape(-1, 3)
            color = np.repeat(color, 3, axis=0).astype(np.float32)
        elif coloring == "vertex":
            color = color.astype(np.float32)
        elif coloring == "uniform":
            assert (type(color) == str)
        return color
    

class Mesh:

    # coloring : str
    # flat_shading :bool
    # num_vertices : np.uint32
    # num_faces : np.uint32
    # T : np.ndarray
    #
    # mouse_over_mesh : bool = False
    # mouse_on_vertex : int = -1
    # vertex_is_draged : bool = False
    #
    # on_transform_hooks = []

    def __init__(self, V, T, color=None, flat_shading=False, out=None, color_wireframe="white", lw_wireframe=1,
                 opacity=1.0, pencil_callback=None):
        """
        NOTE coloring can't be changed
        NOTE if face colors => vertices get duplicated
        # TODO duplicating vertices should be done on client side => go for faces
        """
        self.flat_shading = flat_shading
        self.mouse_over_mesh: bool = False
        self.mouse_on_vertex: int = -1
        self.vertex_is_draged: bool = False
        self.on_transform_hooks: list = []

        self.out = out

        with self.out:

            self.num_vertices = V.shape[0]
            self.num_faces = T.shape[0]
            self.T = T
            self.flat_shading = flat_shading

            self.opacity = opacity

            if color is None:
                color = "white"

            self.coloring = self.get_coloring(color)
            color = self.get_color(color, self.coloring)
            vertices = self.get_vertices(V, self.coloring)
            faces = self.get_faces(T, self.coloring)

            # geometry & material
            if self.coloring == "uniform":
                buffer_geom = py3.BufferGeometry(attributes=dict(
                    position=py3.BufferAttribute(vertices, normalized=False),
                    index=py3.BufferAttribute(faces, normalized=False),
                ))
                material = py3.MeshPhongMaterial(color=color, side="DoubleSide", flatShading=flat_shading,
                                                 opacity=opacity, transparent=True)
            elif self.coloring in ["face", "vertex", "per_edge_per_face"]:
                buffer_geom = py3.BufferGeometry(attributes=dict(
                    position=py3.BufferAttribute(vertices, normalized=False),
                    index=py3.BufferAttribute(faces, normalized=False),
                    color=py3.BufferAttribute(color, normalized=False),
                ))
                material = py3.MeshPhongMaterial(vertexColors='VertexColors', side="DoubleSide",
                                                 flatShading=flat_shading, opacity=opacity, transparent=True)

            if not self.flat_shading:
                buffer_geom.exec_three_obj_method('computeVertexNormals')

            self.mesh = py3.Mesh(
                geometry=buffer_geom,
                material=material
            )

            # wireframe
            self.E = igl.edges(self.T.astype(np.int32))
            self.wireframe = LineSegments(V[self.E, :], color=color_wireframe, linewidth=lw_wireframe)

            # mesh edit interactions
            self.scatter = Scatter(V, out)

            # drawing
            self.drawer = py3.Picker(controlling=self.mesh, event='click')
            self.hover_point = py3.Mesh(geometry=py3.SphereGeometry(radius=0.05),
                                        material=py3.MeshLambertMaterial(color='hotpink'))
            self.hover_point.visible = False
            # self.mesh.add()
            jslink((self.scatter.pencil, 'position'), (self.drawer, 'point'))

            if pencil_callback is not None:
                self.drawer.observe(pencil_callback, names=["point"])
            # self.drawer.observe(self.scatter.on_select, names=["point"])
            # self.drawer.observe(self.picker_on_change, names=["point"])

            self.plane_transform_geom = py3.PlaneGeometry()
            self.plane_transform_mat = py3.MeshPhongMaterial(side="DoubleSide")
            self.plane_transform = py3.Mesh(self.plane_transform_geom, self.plane_transform_mat)
            self.plane_normal = py3.ArrowHelper()
            self.ray_dir = py3.ArrowHelper(color="green")
            self.plane_transform.visible = False
            self.plane_normal.visible = False
            self.ray_dir.visible = False

            self.objects = [self.mesh, self.hover_point, self.plane_transform, self.plane_normal,
                            self.ray_dir] + self.scatter.objects + self.wireframe.objects
            self.controls = [self.drawer] + self.scatter.controls + self.wireframe.controls
            self.keymaps = [] + self.scatter.keymaps + self.wireframe.keymaps
            # self.objects = [self.mesh, self.hover_point]
            # self.controls = [self.drawer]
            # self.keymaps = [] + self.scatter.keymaps

    def handle_event(self, event, keys_pressed, info):
        # with self.out:
        #     print("here")
        info.value = "hi"  # event["event"] + str(keys_pressed)
        if event["event"] == "mousedown":
            if len(keys_pressed) == 2 and "Control" in keys_pressed and "d" in keys_pressed:
                self.hover_point.visible = True
        if event["event"] == "keyup" or event["event"] == "keydown":
            if not (len(keys_pressed) == 2 and "Control" in keys_pressed and "d" in keys_pressed):
                self.hover_point.visible = False
        if event["event"] == "mouseup":
            self.hover_point.visible = False

    def picker_on_change(self, change):
        with self.out:
            if self.vertex_is_draged:
                return
            owner = change["owner"]
            # print(change)
            if owner.object is not None:
                self.mouse_over_mesh = True
                point = np.array(change["new"])
                tmp = self.scatter.mouse_over_vertex(point, self.scatter.radius_select * self.scatter.radius_scale)
                if tmp != -1:
                    # print("here", self.mouse_on_vertex, tmp)
                    self.scatter.highlight_vertex(tmp, self.scatter.color_highlight)
                elif self.mouse_on_vertex != -1:  # tmp == -1 => reset color
                    self.scatter.highlight_vertex(self.mouse_on_vertex, self.scatter.color_default)
                self.mouse_on_vertex = tmp
            else:
                # print("there")
                self.mouse_over_mesh = False
                if self.mouse_on_vertex != -1:
                    self.scatter.highlight_vertex(self.mouse_on_vertex, self.scatter.color_default)
                self.mouse_on_vertex = -1

    def soft_vertex_transform(self, mouse_event, mouse_is_down, control_state):
        if not control_state.viewer.camera_control.enableRotate:
            with self.out:
                if mouse_event["event"] == "mouseup":
                    # print("mouseup")
                    if self.vertex_is_draged:
                        mx = mouse_event["relativeX"]
                        my = mouse_event["relativeY"]
                        pt_on_sensor = utils_viewer.mouse2point(mx, my, control_state.viewer.renderer,
                                                                control_state.viewer.camera)
                        ray_origin = np.array(control_state.viewer.camera.position)
                        ray_dir = self._py_quat.rotate(pt_on_sensor)
                        ray_dir /= np.linalg.norm(ray_dir)
                        pt_on_plane = utils_viewer.ray_plane_intersection(self._plane_origin, self._plane_normal,
                                                                          ray_origin, ray_dir)
                        for on_transform in self.on_transform_hooks:
                            on_transform(self.mouse_on_vertex, pt_on_plane, True)
                        self.ray_dir.origin = (pt_on_plane).tolist()
                        self.ray_dir.dir = (-ray_dir).tolist()

                        self.vertex_is_draged = False
                        self.scatter.highlight_vertex(self.mouse_on_vertex, self.scatter.color_highlight)
                        # NOTE => Bug in Picker
                        self.mouse_over_mesh = False

                elif mouse_event["event"] == "mousemove":
                    if self.vertex_is_draged:
                        mx = mouse_event["relativeX"]
                        my = mouse_event["relativeY"]
                        pt_on_sensor = utils_viewer.mouse2point(mx, my, control_state.viewer.renderer,
                                                                control_state.viewer.camera)
                        ray_origin = np.array(control_state.viewer.camera.position)
                        ray_dir = self._py_quat.rotate(pt_on_sensor)
                        ray_dir /= np.linalg.norm(ray_dir)
                        pt_on_plane = utils_viewer.ray_plane_intersection(self._plane_origin, self._plane_normal,
                                                                          ray_origin, ray_dir)
                        self.ray_dir.origin = (pt_on_plane).tolist()
                        self.ray_dir.dir = (-ray_dir).tolist()
                        for on_transform in self.on_transform_hooks:
                            on_transform(self.mouse_on_vertex, pt_on_plane)
                    else:
                        # NOTE => Bug in Picker
                        if not self.mouse_over_mesh:
                            if self.mouse_on_vertex != -1:
                                self.scatter.highlight_vertex(self.mouse_on_vertex, self.scatter.color_default)
                                self.mouse_on_vertex = -1

                    # else:
                    #     print(control_state.viewer.camera_control.get_state())

                elif mouse_event["event"] == "mousedown":
                    if self.mouse_on_vertex != -1:
                        # print("mousedown")
                        # print(control_state.viewer.camera_control.get_state())
                        self.vertex_is_draged = True
                        # control_state.viewer.toggle_camera_control()
                        self.scatter.highlight_vertex(self.mouse_on_vertex, self.scatter.color_selected)
                        # TODO compute plane
                        self._plane_origin = self.scatter.point_group.geometry.attributes["position"].array[
                                             self.mouse_on_vertex, :]
                        quat = control_state.viewer.camera.quaternion
                        self._py_quat = pq.Quaternion(quat[3], quat[0], quat[1], quat[2])
                        self._plane_normal = utils_viewer.camera_lookat_world((quat[3], quat[0], quat[1], quat[2]))
                        # Visualize plane
                        self.plane_transform.position = self._plane_origin.tolist()
                        self.plane_transform.quaternion = quat
                        self.plane_normal.origin = self._plane_origin.tolist()
                        self.plane_normal.dir = self._plane_normal.tolist()
                        # Visualize ray
                        mx = mouse_event["relativeX"]
                        my = mouse_event["relativeY"]
                        pt_on_sensor = utils_viewer.mouse2point(mx, my, control_state.viewer.renderer,
                                                                control_state.viewer.camera)
                        ray_origin = np.array(control_state.viewer.camera.position)
                        ray_dir = self._py_quat.rotate(pt_on_sensor)
                        ray_dir /= np.linalg.norm(ray_dir)
                        # print("dot:", np.dot(self._plane_normal, ray_dir))
                        pt_on_plane = utils_viewer.ray_plane_intersection(self._plane_origin, self._plane_normal,
                                                                          ray_origin, ray_dir)
                        # print(pt_on_sensor, self._plane_origin, pt_on_plane)
                        self.ray_dir.origin = (pt_on_plane).tolist()
                        self.ray_dir.dir = (-ray_dir).tolist()

                    # else:
                    # print("mousedown")
                    # control_state.viewer.camera_control.enableRotate = True

    def update_in_draw_state(self, is_active, control_state):

        pass

    def update(self, V=None, color=None):
        if V is not None:
            vertices = self.get_vertices(V, self.coloring)
            self.update_vertices(vertices)
            self.scatter.update(V)
            self.wireframe.update(V[self.E, :])
        if color is not None:
            with self.out:
                color = self.get_color(color, self.coloring)
                self.update_color(color, self.coloring)

    def update_faces(self, faces):
        self.mesh.geometry.attributes["index"].array = faces
        self.mesh.geometry.attributes["index"].needsUpdate = True

    def update_vertices(self, vertices):
        self.mesh.geometry.attributes["position"].array = vertices
        self.mesh.geometry.attributes["position"].needsUpdate = True
        self.mesh.geometry.exec_three_obj_method("computeBoundingSphere")

    def update_color(self, color, coloring):
        if coloring == "uniform":
            self.mesh.material.color = color
        elif coloring in ["vertex", "face", "per_edge_per_face"]:
            self.mesh.geometry.attributes["color"].array = color
            self.mesh.geometry.attributes["color"].needsUpdate = True

    def get_coloring(self, color):
        if type(color) == str:
            coloring = "uniform"
        elif type(color) == np.ndarray:
            if len(color.shape) == 2 and color.shape[0] == self.num_faces:
                coloring = "face"
            elif color.shape[0] == self.num_vertices:
                coloring = "vertex"
            elif len(color.shape) == 3 and color.shape[0] == self.num_faces:
                coloring = "per_edge_per_face"
            else:
                coloring = False
        else:
            coloring = False
        return coloring

    def get_faces(self, T, coloring):
        if coloring == "face":
            faces = np.linspace(0, T.shape[0] * 3 - 1, T.shape[0] * 3, dtype=np.uint16)
        elif coloring == "per_edge_per_face":
            faces = np.linspace(0, T.shape[0] * 3 * 3 - 1, T.shape[0] * 3 * 3, dtype=np.uint16)
        elif coloring in ["uniform", "vertex"]:
            faces = T.flatten().astype(np.uint16)
        return faces

    def get_vertices(self, V, coloring):
        if coloring == "face":
            vertices = V[self.T.flatten(), :].astype(np.float32)
        elif coloring == "per_edge_per_face":
            V_epi = np.sum(V[self.T, :], axis=1) / 3
            vertices = np.ndarray((self.T.shape[0] * 3 * 3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(2):
                    v = (i + j) % 3
                    vertices[i * 3 + j::9, :] = V[self.T[:, v], :]
                j = 2
                vertices[i * 3 + j::9, :] = V_epi
            vertices = vertices.astype(np.float32)
        elif coloring in ["uniform", "vertex"]:
            vertices = V.astype(np.float32)
        return vertices

    def get_color(self, color, coloring):
        if coloring == "face":
            color = np.repeat(color, 3, axis=0).astype(np.float32)
        elif coloring == "per_edge_per_face":
            color = color.reshape(-1, 3)
            color = np.repeat(color, 3, axis=0).astype(np.float32)
        elif coloring == "vertex":
            color = color.astype(np.float32)
        elif coloring == "uniform":
            assert (type(color) == str)
        return color

    def draw(self):
        pass


class LineSegments:
    # num_vertices : int
    # num_edges: int
    # coloring: str

    def __init__(self, V, color=None, linewidth=1):

        # if type(T) == np.ndarray:
        #     E = igl.edges(T.astype(np.int32))

        # sequential vertices
        # TODO there is class for sequential lines
        # if len(V.shape) == 2 and type(E) == None:
        #     positions = np.stack([V[:-1,:], V[1:,:]],axis=1).astype(np.float32)
        # # vertices with edge tensor
        # elif len(V.shape) == 2 and type(E) == np.ndarray:
        #     positions = V[E,:].astype(np.float32)

        self.num_edges = V.shape[0]
        # self.num_vertices = V.shape[0]

        if color is None:
            color = "white"

        # per vertex/edge color
        # if type(color) == np.ndarray:
        #     # per edge function values
        #     if len(color.shape) == 1:
        #         color = get_colors(color)

        #     # per edge colors
        #     if len(color.shape) == 2 and color.shape[1] == 3:
        #         color = np.stack([color, color], axis = 1)

        #     # per vertex function values
        #     if len(color.shape) == 2 and color.shape[1] == 2:
        #         color = np.stack([get_colors(color[:,0]),get_colors(color[:,1])], axis = 1)

        #     line_segment_colors = color

        self.coloring = self.get_coloring(color)
        color = self.get_color(color, self.coloring)
        positions = self.get_vertices(V, self.coloring)

        # geometry & material
        if type(color) == str:
            line_segment_geom = py3.LineSegmentsGeometry(
                positions=positions,
            )
            line_material = py3.LineMaterial(linewidth=linewidth, color=color)
        elif type(color) == np.ndarray:
            line_segment_geom = py3.LineSegmentsGeometry(
                positions=positions,
                colors=color,
            )
            line_material = py3.LineMaterial(linewidth=linewidth, vertexColors='VertexColors')

        self.line_segments = py3.LineSegments2(line_segment_geom, line_material)

        self.objects = [self.line_segments]
        self.controls = []
        self.keymaps = []

    def toggle(self):
        self.line_segments.visible = not self.line_segments.visible

    def update(self, V=None, color=None):

        if V is not None:
            positions = self.get_vertices(V, self.coloring)
            self.update_vertices(positions)

        if color is not None:
            color = self.get_color(color, self.coloring)
            self.update_color(color, self.coloring)

    def update_vertices(self, positions):
        # self.mesh.geometry.attributes["position"].array = vertices
        # self.mesh.geometry.attributes["position"].needsUpdate = True
        self.line_segments.geometry.positions = positions
        self.line_segments.geometry.exec_three_obj_method("computeBoundingSphere")

    def update_color(self, color, coloring):
        if coloring == "uniform":
            self.line_segments.material.color = color
        elif coloring in ["vertex", "edge"]:
            # self.mesh.geometry.attributes["color"].array = color
            # self.mesh.geometry.attributes["color"].needsUpdate = True
            self.line_segments.geometry.colors = color

    def get_coloring(self, color):
        if type(color) == str:
            coloring = "uniform"
        elif type(color) == np.ndarray:
            if color.shape[1] == 3:
                coloring = "edge"
            elif color.shape[1] == 2 and color.shape[2] == 3:
                coloring = "vertex"
            else:
                coloring = False
        else:
            coloring = False
        return coloring

    def get_vertices(self, V, coloring):
        vertices = V.astype(np.float32)
        return vertices

    def get_color(self, color, coloring):
        if coloring == "edge":
            # color = np.repeat(color, 2, axis=0).astype(np.float32)
            color = np.stack([color] * 2, axis=1).astype(np.float32)
        elif coloring == "vertex":
            color = color.astype(np.float32)
        elif coloring == "uniform":
            assert (type(color) == str)
        return color


# Helper functions
def get_colors(inp, colormap="viridis", normalize=True, vmin=None, vmax=None):
    colormap = plt.cm.get_cmap(colormap)
    if normalize:
        vmin = np.min(inp)
        vmax = np.max(inp)

    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))[:, :3]
