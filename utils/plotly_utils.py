import plotly.graph_objects as go
import numpy as np


def plot_mesh(verts, faces):
    """
    Plot a mesh using plotly.

    Parameters:
    - verts: np.ndarray with shape (n, 3), representing the n vertices.
    - faces: np.ndarray with shape (m, 3), representing the m faces.
    """
    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=verts[:,0],
            y=verts[:,1],
            z=verts[:,2],
            # i, j and k give the vertices of triangles
            i = faces[:, 0],
            j = faces[:, 1],
            k = faces[:, 2],
            name='y',
            showscale=True
        )
    ])

    fig.update_layout(
        autosize=True,
        width=800,
        height=600,
        # scene_aspectmode='cube'
        scene_aspectmode='data'
    )

    fig.show()


def plot_scatter(points: list[np.ndarray], sizes: list[int] = None, colors: list[np.ndarray] = None):
    """
    Plot a scatter plot using plotly.

    Parameters:
    - points: list of np.ndarray with shape (n, 3), representing the n points.
    """
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[i][:,0],
            y=points[i][:,1],
            z=points[i][:,2],
            mode='markers',
            marker=dict(
                size=sizes[i] if sizes is not None else 2,
                color=colors[i] if colors is not None else np.ones((points[i].shape[0], 3)) * np.array([0, 0, i / len(points)]),
                opacity=0.8
            )
        ) for i in range(len(points))
    ])

    fig.update_layout(
        autosize=True,
        width=800,
        height=600,
        # scene_aspectmode='cube'
        scene_aspectmode='data'
    )

    fig.show()