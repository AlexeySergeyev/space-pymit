import numpy as np


def _parse_minkowski_output(output: str) -> tuple[np.ndarray, list[list[int]]]:
    """
    Parse the standard output of the minkowski fortran binary.

    Format:
    num_vertices, num_faces
    v1_x v1_y v1_z
    ...
    [for each face]
    num_vertices_in_face
    idx1 idx2 idx3 ... (1-based indices)
    """
    lines = output.strip().splitlines()
    if not lines:
        raise ValueError("Empty output from minkowski.")

    try:
        parts = lines[0].split()
        num_vertices = int(parts[0])
        num_faces = int(parts[1])
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Failed to parse header 'num_vertices num_faces' from: {lines[0]}"
        ) from e

    v_lines = lines[1 : num_vertices + 1]
    f_lines = lines[num_vertices + 1 :]

    vertices = []
    for line in v_lines:
        coords = [float(x) for x in line.split()]
        if len(coords) != 3:
            raise ValueError(f"Invalid vertex line: {line}")
        vertices.append(coords)

    faces = []
    idx = 0
    while idx < len(f_lines):
        if not f_lines[idx].strip():
            idx += 1
            continue

        num_verts_in_face = int(f_lines[idx].strip())
        idx += 1

        face_indices_str = f_lines[idx].strip().split()
        face_indices = [int(v) for v in face_indices_str]
        if len(face_indices) != num_verts_in_face:
            raise ValueError(
                f"Expected {num_verts_in_face} vertices for face, got {len(face_indices)}: {f_lines[idx]}"
            )

        faces.append(face_indices)
        idx += 1

    if len(faces) != num_faces:
        raise ValueError(f"Expected {num_faces} faces, got {len(faces)}.")

    return np.array(vertices), faces


def _triangulate_faces(faces: list[list[int]]) -> list[tuple[int, int, int]]:
    """
    Convert arbitrary polygonal faces to triangles.

    Returns 0-based index tuples.
    """
    triangles = []
    for face in faces:
        face_0 = [idx - 1 for idx in face]
        v0 = face_0[0]
        for i in range(1, len(face_0) - 1):
            triangles.append((v0, face_0[i], face_0[i + 1]))

    return triangles


def load_model_obj(file_path: str) -> tuple[np.ndarray, list[list[int]]]:
    """Load an asteroid 3D shape model from a standard Wavefront .obj file."""
    vertices = []
    faces = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.strip().split()
            if parts[0] == "v":
                if len(parts) >= 4:
                    coords = [float(p) for p in parts[1:4]]
                    vertices.append(coords)
            elif parts[0] == "f":
                face_indices = []
                for p in parts[1:]:
                    v_idx_str = p.split("/")[0]
                    face_indices.append(int(v_idx_str))
                faces.append(face_indices)

    return np.array(vertices), faces


def save_model_obj(
    vertices: np.ndarray, faces: list[list[int]], save_path: str
) -> None:
    """Save the 3D model in Wavefront .obj format."""
    with open(save_path, "w") as f:
        f.write("# Asteroid Shape Model\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")
        for face in faces:
            face_str = " ".join(str(idx) for idx in face)
            f.write(f"f {face_str}\n")
