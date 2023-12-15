
import numpy as np
import pyquaternion as pqt
import igl

# TODO should be static class, container for functions
class Soft_Transform:

    # V : np.ndarray
    # T : np.ndarray
    # E : np.ndarray
    #
    # n_v : np.int32
    # indices : np.ndarray
    #
    # vv = None
    # vt = None

    radius : np.float64 = 1.0
    std : np.float64 = 1.0 / 3
    direction : np.ndarray
    step_size : np.float64 = 0.1

    # dir_scale : np.float64 = None

    # vid = None

    # weighting = "gauss"
    # weighting_options = {}

    def __init__(self):
        # self.V = V
        # self.dir_scale = np.max(np.linalg.norm(self.V[self.E[:,1]] - self.V[self.E[:,0]], axis=-1))
        # self.n_v = V.shape[0]

        self.n_v = -1

        self.T = None
        self.E = None

        self.vv = []
        self.vt = []

        self.weighting = "gauss"
        self.weighting_options = {}
        self.weighting_options["gauss"] = self.gauss
        self.weighting_options["laplace"] = self.laplace

    def init_indices(self, T):
        self.n_v = np.max(T)+1

        self.T = T
        self.E = igl.edges(T)

        self.vv = []
        self.vt = []
        for i in range(self.n_v):
            self.vv.append([])
            self.vt.append([])
        for e in range(self.E.shape[0]):
            v1 = self.E[e,0]; v2 = self.E[e,1]
            self.vv[v1].append(v2)
            self.vv[v2].append(v1)
        for t in range(self.T.shape[0]):
            for v in T[t]:
                self.vt[v].append(t)

        self.indices = np.linspace(0,self.n_v-1, self.n_v, dtype=np.int32)



    def gauss(self, x, s, mu = 0):
        """
        Args:
            x : should be the distances
            mu : should be zero
            s : scalar (should be radius)
        """
        return 1 / (s * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / s)**2)

    def laplace(self, x, b, mu = 0):
        return 1 / (2*b) * np.exp(-np.abs((x - mu)/ b))


    def transform(self, V, vid, v_coords):
        self.V = V.copy()
        self.vid = vid
        step = v_coords - V[vid]
        out, distances = self.get_pts_in_radius()
        indices = self.indices[out]
        distances = distances[out]
        if self.weighting in self.weighting_options:
            weights = self.weighting_options[self.weighting](distances, self.std)
            weights /= np.max(weights)
            tmp = np.expand_dims(weights,axis=-1) * np.repeat(step.reshape(1,3), repeats=weights.shape[0],axis=0)
            # print(tmp.shape)
            self.V[indices,:] += tmp
        return self.V


    def make_deform_step(self, step_size):
        if self.vid is not None:
            out, distances = self.get_pts_in_radius()
            indices = self.indices[out]
            distances = distances[out]
            if self.weighting in self.weighting_options:
                weights = self.weighting_options[self.weighting](distances, self.std)
                tmp = step_size * np.expand_dims(weights,axis=-1) * np.repeat(self.direction.reshape(1,3), repeats=weights.shape[0],axis=0)
                # print(tmp.shape)
                self.V[indices,:] += tmp

    def update_direction(self, a_normal, a_ortho):
        """
        Args:
            a_normal : angle with rot axis = normal
            a_orth : angle with rot axis orthogonal to normal and e1
        """
        
        vertex_normals = igl.per_vertex_normals(self.V, self.T)
        if self.vid is not None:
            t0 = self.vt[self.vid][0]
            e01 = self.V[self.T[t0,1]] - self.V[self.T[t0,0]]

            normal = vertex_normals[self.vid]
            ortho = np.cross(normal, e01)

            p1 = pqt.Quaternion(axis=normal, angle=a_normal)
            p2 = pqt.Quaternion(axis=ortho, angle=a_ortho)

            self.direction = self.dir_scale * p2.rotate(normal)
            self.direction = p1.rotate(self.direction)

    def get_pts_in_radius(self):
        out = np.full(self.n_v, False)
        distances = np.full(self.n_v, np.inf)
        distances[self.vid] = 0
        self.dijkstra(self.vid, self.radius, out, distances)
        return out, distances

    def dijkstra(self, vid, radius, out, distances):
        # get neighbours
        neighbours = self.vv[vid]

        # self.vid won't be visited anymore
        out[vid] = True

        # update distances
        egde_lenghts = np.linalg.norm(self.V[neighbours,:] - self.V[vid,:], axis=-1)
        distances_tmp = distances[vid] + egde_lenghts
        mask = distances_tmp > distances[neighbours]
        distances_tmp[mask] = distances[neighbours][mask]
        distances[neighbours] = distances_tmp  # NOTE for some reason distances[neighbours][mask] is not possible

        # select vertex with smallest distance that is not in out
        unvisited = np.logical_not(out)
        vid_next = self.indices[unvisited][np.argmin(distances[unvisited])]

        # print(vid_next)
        # print(distances)

        if distances[vid_next] <= radius:
            self.dijkstra(vid_next, radius, out, distances)



        


