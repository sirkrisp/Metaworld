import mujoco
import numpy as np

class EnvRenderer:

    def __init__(self, env, camera_name, height=240, width: int = 320) -> None:
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.env = env

    def render(self, camera_name=None, depth=False, segmentation=False):
        # mujoco.mj_forward(self.env.model, self.env.data)
        self.renderer.update_scene(self.env.data, self.camera_name if camera_name is None else camera_name)
        img = self.renderer.render()
        data = {"img": img}
        if depth:
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            data["depth"] = depth
            self.renderer.disable_depth_rendering()
        if segmentation:
            # NOTE seg[:,:,0] is the geom id, seg[:,:,1] is the object type.
            # Furthermore, if the object is a site, then the geom id is the one of the geom behind the site!
            self.renderer.enable_segmentation_rendering()
            seg = self.renderer.render()
            data["seg"] = seg
            self.renderer.disable_segmentation_rendering()
        return data
    
def get_random_color(seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(3)
    
def process_seg_data(seg: np.ndarray):
    # Display the contents of the first channel, which contains object
    # IDs. The second channel, seg[:, :, 1], contains object types.
    # Infinity is mapped to -1
    geom_ids = seg[:, :, 0]
    geom_id_list = list(set(geom_ids.flatten().tolist()))
    rgb_pixels = np.zeros((seg.shape[0], seg.shape[1], 3))
    for i, geom_id in enumerate(geom_id_list):
        rgb_pixels[geom_ids == geom_id, :] = get_random_color(i * 23)
    return rgb_pixels