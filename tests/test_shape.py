import tempfile
import unittest
from pathlib import Path

import numpy as np

from pymit.shape import (
    _parse_minkowski_output,
    _triangulate_faces,
    load_model_obj,
    save_model_obj,
)


class ShapeParsingTests(unittest.TestCase):
    def test_parse_minkowski_output_reads_vertices_and_faces(self):
        output = """4 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
3
1 2 3
3
1 2 4
"""

        vertices, faces = _parse_minkowski_output(output)

        np.testing.assert_allclose(
            vertices,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
        self.assertEqual(faces, [[1, 2, 3], [1, 2, 4]])

    def test_triangulate_faces_converts_polygons_to_zero_based_triangles(self):
        triangles = _triangulate_faces([[1, 2, 3, 4]])

        self.assertEqual(triangles, [(0, 1, 2), (0, 2, 3)])


class ObjFileTests(unittest.TestCase):
    def test_save_and_load_model_obj_round_trip(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = [[1, 2, 3]]

        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "model.obj"

            save_model_obj(vertices, faces, str(output_file))
            loaded_vertices, loaded_faces = load_model_obj(str(output_file))

        np.testing.assert_allclose(loaded_vertices, vertices)
        self.assertEqual(loaded_faces, faces)


if __name__ == "__main__":
    unittest.main()
