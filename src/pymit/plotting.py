from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .shape import _triangulate_faces


def plot_model(
    vertices: np.ndarray,
    faces: list[list[int]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize the 3D asteroid shape model with Matplotlib."""
    triangles = _triangulate_faces(faces)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    tri_coords = vertices[[list(tri) for tri in triangles]]
    collection = Poly3DCollection(
        tri_coords, facecolors="lightgray", linewidths=0.5, edgecolors="k", alpha=0.9
    )
    ax.add_collection3d(collection)

    all_x = vertices[:, 0]
    all_y = vertices[:, 1]
    all_z = vertices[:, 2]

    max_range = (
        np.array(
            [
                all_x.max() - all_x.min(),
                all_y.max() - all_y.min(),
                all_z.max() - all_z.min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Asteroid Shape Model")
    ax.view_init(elev=30, azim=45)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_model_plotly(
    vertices: np.ndarray,
    faces: list[list[int]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize the 3D asteroid shape model using Plotly."""
    triangles = _triangulate_faces(faces)

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[tri[0] for tri in triangles],
                j=[tri[1] for tri in triangles],
                k=[tri[2] for tri in triangles],
                color="lightgray",
                opacity=1.0,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
                lightposition=dict(x=100, y=100, z=100),
            )
        ]
    )
    fig.update_layout(
        title="Interactive Asteroid Shape Model",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_path:
        fig.write_html(save_path)

    if show:
        fig.show()
