import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RANSACRegressor


def get_close_feature_points(query_world_coord: torch.Tensor, world_coords: torch.Tensor, ignore: torch.Tensor, n_closest=1):
    """ Get closest feature points to query_world_coord
    Args:
        query_world_coord (torch.Tensor): (3,) tensor of query world coordinate
        world_coords (torch.Tensor): (N, 3) tensor of world coordinates
        ignore (torch.Tensor): (N,) tensor of bools to ignore some points
        n_closest (int, optional): number of closest points to return. Defaults to 1.
    Returns:
        closes_point_indices (torch.Tensor): (n_closest,) tensor of indices of closest points
    """
    # compute distance
    dist = torch.norm(world_coords - query_world_coord, dim=1)
    # ignore
    dist[ignore] = torch.tensor(float("inf"))
    # get closest points
    _, closest_point_indices = torch.topk(dist, k=n_closest, largest=False)
    return closest_point_indices

def get_close_feature_points_np(query_world_coord: np.ndarray, world_coords: np.ndarray, ignore = None, n_closest=1):
    """ Get closest feature points to query_world_coord
    Args:
        query_world_coord (np.ndarray): (3,) array of query world coordinate
        world_coords (np.ndarray): (N, 3) array of world coordinates
        ignore (np.ndarray): (N,) array of bools to ignore some points
        n_closest (int, optional): number of closest points to return. Defaults to 1.
    Returns:
        closest_point_indices (np.ndarray): (n_closest,) array of indices of closest points
    """
    # compute distance
    dist = np.linalg.norm(world_coords - query_world_coord, axis=1)
    # ignore
    if ignore is not None:
        dist[ignore] = np.inf
    # get closest points
    closest_point_indices = np.argpartition(dist, n_closest)[:n_closest]
    return closest_point_indices


# Define a custom base estimator for rotation and translation
class RigidTransformEstimator(BaseEstimator, TransformerMixin):
    def fit(self, X, Y):
        # Calculate the centroid of both point clouds
        centroid_X = np.mean(X, axis=0)
        centroid_Y = np.mean(Y, axis=0)

        # Compute the centered data points
        X_centered = X - centroid_X
        Y_centered = Y - centroid_Y

        # Calculate the covariance matrix
        covariance_matrix = np.dot(X_centered.T, Y_centered)

        # Perform SVD to get the rotation matrix
        U, S, Vt = np.linalg.svd(covariance_matrix)
        rotation_matrix = np.dot(U, Vt)

        # Compute the translation vector
        translation_vector = centroid_Y - np.dot(centroid_X, rotation_matrix)
        self.translation_vector_no_rot = centroid_Y - centroid_X

        self.rotation_ = rotation_matrix
        self.translation_ = translation_vector
        return self

    def transform(self, X):
        # return np.dot(X, self.rotation_) + self.translation_
        return np.dot(X, self.rotation_) + self.translation_
    
    def score(self, X, Y):
        return np.mean(np.linalg.norm(self.transform(X) - Y, axis=1))
    
    def predict(self, X):
        return self.transform(X)

# Create a RANSAC model with the custom base estimator for rotation and translation estimation
ransac = RANSACRegressor(estimator=RigidTransformEstimator(), min_samples=3, residual_threshold=0.1)

def compute_T(X, Y):
    ransac.fit(X, Y)
    R, t, t_no_rot = np.array(ransac.estimator_.rotation_), np.array(ransac.estimator_.translation_), np.array(ransac.estimator_.translation_vector_no_rot)
    x_center = np.mean(X, axis=0)
    def predict(X_new):
        return np.dot(X_new - x_center, R) + t_no_rot + x_center
    return R, t, t_no_rot, x_center, predict


# Define a custom base estimator for rotation and translation
class TranslationEstimatorXY(BaseEstimator, TransformerMixin):
    def fit(self, X, Y):
        # Calculate the centroid of both point clouds
        centroid_X = np.mean(X, axis=0)
        centroid_Y = np.mean(Y, axis=0)

        self.rotation_ = np.eye(3)
        self.translation_ = centroid_Y - centroid_X
        self.translation_[2] = 0
        return self

    def transform(self, X):
        return X + self.translation_
    
    def score(self, X, Y):
        return np.mean(np.linalg.norm(self.transform(X) - Y, axis=1))
    
    def predict(self, X):
        return self.transform(X)
    
ransac_translation = RANSACRegressor(estimator=TranslationEstimatorXY(), min_samples=2, residual_threshold=0.1)

def estimate_translation(X, Y):
    ransac_translation.fit(X, Y)
    t = np.array(ransac_translation.estimator_.translation_)
    def predict(X_new):
        return X_new + t
    return t, predict