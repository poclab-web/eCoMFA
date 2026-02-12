"""Helpers to render CoMFA grids, axes, and geometric annotations in 3D views."""

import itertools

import numpy as np


def addmesh(xyzview):
    """
    Add a cubic CoMFA reference mesh to a py3Dmol-like viewer object.

    The mesh consists of edge lines and face guide lines in the same coordinate
    system used by the CoMFA grid.
    """
    radius = 0.003
    step = 0.529177 * 2
    ranges = np.array([[-12, 6], [-6, 6], [-12, 12]]) * 0.529177

    def addcylinder_func(line_ranges, view, cylinder_radius):
        view.addCylinder(
            {
                "start": {"x": line_ranges[0][0], "y": line_ranges[1][0], "z": line_ranges[2][0]},
                "end": {"x": line_ranges[0][1], "y": line_ranges[1][1], "z": line_ranges[2][1]},
                "radius": cylinder_radius,
                "color": "gray",
            }
        )
        return view

    # Optional inner grid lines.
    if False:
        for x, y in itertools.product(range(ranges[0][0] + 1, ranges[0][1]), range(ranges[1][0] + 1, ranges[1][1])):
            xyzview = addcylinder_func([[x, x], [y, y], ranges[2]], xyzview, radius)
        for y, z in itertools.product(range(ranges[1][0] + 1, ranges[1][1]), range(ranges[2][0] + 1, ranges[2][1])):
            xyzview = addcylinder_func([ranges[0], [y, y], [z, z]], xyzview, radius)
        for z, x in itertools.product(range(ranges[2][0] + 1, ranges[2][1]), range(ranges[0][0] + 1, ranges[0][1])):
            xyzview = addcylinder_func([[x, x], ranges[1], [z, z]], xyzview, radius)

    # Grid edges.
    if True:
        for x, y in itertools.product(ranges[0], ranges[1]):
            xyzview = addcylinder_func([[x, x], [y, y], ranges[2]], xyzview, radius * 5)

        for y, z in itertools.product(ranges[1], ranges[2]):
            xyzview = addcylinder_func([ranges[0], [y, y], [z, z]], xyzview, radius * 5)

        for z, x in itertools.product(ranges[2], ranges[0]):
            xyzview = addcylinder_func([[x, x], ranges[1], [z, z]], xyzview, radius * 5)

    # Face guide lines.
    if True:
        lines = list(itertools.product(ranges[0], np.arange(ranges[1][0], ranges[1][1] + step, step))) + list(
            itertools.product(np.arange(ranges[0][0], ranges[0][1] + step, step), ranges[1])
        )
        for x, y in lines:
            if x == ranges[0][1] or y == ranges[1][1]:
                xyzview = addcylinder_func([[x, x], [y, y], ranges[2]], xyzview, radius * 2)

        lines = list(itertools.product(ranges[1], np.arange(ranges[2][0], ranges[2][1] + step, step))) + list(
            itertools.product(np.arange(ranges[1][0], ranges[1][1] + step, step), ranges[2])
        )
        for y, z in lines:
            if y == ranges[1][1] or z == ranges[2][1]:
                xyzview = addcylinder_func([ranges[0], [y, y], [z, z]], xyzview, radius * 2)

        lines = list(itertools.product(ranges[2], np.arange(ranges[0][0], ranges[0][1] + step, step))) + list(
            itertools.product(np.arange(ranges[2][0], ranges[2][1] + step, step), ranges[0])
        )
        for z, x in lines:
            if z == ranges[2][1] or x == ranges[0][1]:
                xyzview = addcylinder_func([[x, x], ranges[1], [z, z]], xyzview, radius * 2)



def addxyzarrow(xyzview):
    """Add x/y/z axis arrows and labels to a 3D viewer object."""
    axis_length = 4
    radius = 0.025
    color = "gray"

    xyzview.addArrow(
        {
            "start": {"x": -axis_length, "y": 0, "z": 0},
            "end": {"x": axis_length, "y": 0, "z": 0},
            "radius": radius,
            "radiusRatio": 4,
            "mid": 0.9,
            "color": color,
        }
    )
    xyzview.addArrow(
        {
            "start": {"x": 0, "y": -axis_length, "z": 0},
            "end": {"x": 0, "y": axis_length, "z": 0},
            "radius": radius,
            "radiusRatio": 4,
            "mid": 0.9,
            "color": color,
        }
    )
    xyzview.addArrow(
        {
            "start": {"x": 0, "y": 0, "z": -axis_length},
            "end": {"x": 0, "y": 0, "z": axis_length},
            "radius": radius,
            "radiusRatio": 4,
            "mid": 0.9,
            "color": color,
        }
    )

    xyzview.addLabel(
        "x",
        {"position": {"x": axis_length, "y": 0, "z": 0}, "backgroundColor": color, "backgroundOpacity": 0.5},
    )
    xyzview.addLabel(
        "y",
        {"position": {"x": 0, "y": axis_length, "z": 0}, "backgroundColor": color, "backgroundOpacity": 0.5},
    )
    xyzview.addLabel(
        "z",
        {"position": {"x": 0, "y": 0, "z": axis_length}, "backgroundColor": color, "backgroundOpacity": 0.5},
    )



def add_label(xyz, param, center):
    """
    Add geometric annotation arrows/arcs around a center point in the viewer.

    Args:
        xyz: 3D viewer object with `addArrow`, `addCylinder`, and `addSphere` methods.
        param (sequence): Geometry parameters used for arc radius and angle drawing.
        center (sequence): Anchor coordinate `[x, y, z]` for mirrored annotation points.
    """
    radius = 0.05
    radius_ratio = 3
    color = "black"

    xyz.addArrow(
        {
            "start": {"x": 0, "y": 0, "z": 0},
            "end": {"x": center[0], "y": 0, "z": center[2]},
            "radius": radius,
            "color": color,
            "radiusRatio": radius_ratio,
            "mid": 0.8,
        }
    )
    xyz.addArrow(
        {
            "start": {"x": 0, "y": 0, "z": 0},
            "end": {"x": center[0], "y": 0, "z": -center[2]},
            "radius": radius,
            "color": color,
            "radiusRatio": radius_ratio,
            "mid": 0.8,
        }
    )

    xyz.addArrow(
        {
            "start": {"x": center[0] + param[0] / 2, "y": center[1], "z": center[2]},
            "end": {"x": center[0] + param[0], "y": center[1], "z": center[2]},
            "radius": radius,
            "color": color,
            "radiusRatio": radius_ratio,
            "mid": 0.5,
        }
    )
    xyz.addArrow(
        {
            "start": {"x": center[0] + param[0] / 2, "y": center[1], "z": center[2]},
            "end": {"x": center[0], "y": center[1], "z": center[2]},
            "radius": radius,
            "color": color,
            "radiusRatio": radius_ratio,
            "mid": 0.5,
        }
    )
    xyz.addArrow(
        {
            "start": {"x": center[0] + param[0] / 2, "y": center[1], "z": -center[2]},
            "end": {"x": center[0] + param[0], "y": center[1], "z": -center[2]},
            "radius": radius,
            "color": color,
            "radiusRatio": radius_ratio,
            "mid": 0.5,
        }
    )
    xyz.addArrow(
        {
            "start": {"x": center[0] + param[0] / 2, "y": center[1], "z": -center[2]},
            "end": {"x": center[0], "y": center[1], "z": -center[2]},
            "radius": radius,
            "color": color,
            "radiusRatio": radius_ratio,
            "mid": 0.5,
        }
    )

    for angle in np.arange(0, param[2], 1):
        xyz.addCylinder(
            {
                "start": {
                    "x": param[1] / 2 * np.cos(np.radians(angle)),
                    "y": 0,
                    "z": param[1] / 2 * np.sin(np.radians(angle)),
                },
                "end": {
                    "x": param[1] / 2 * np.cos(np.radians(angle + 1.1)),
                    "y": 0,
                    "z": param[1] / 2 * np.sin(np.radians(angle + 1.1)),
                },
                "radius": radius,
                "color": color,
            }
        )
        xyz.addCylinder(
            {
                "start": {
                    "x": param[1] / 2 * np.cos(np.radians(angle)),
                    "y": 0,
                    "z": -param[1] / 2 * np.sin(np.radians(angle)),
                },
                "end": {
                    "x": param[1] / 2 * np.cos(np.radians(angle + 1.1)),
                    "y": 0,
                    "z": -param[1] / 2 * np.sin(np.radians(angle + 1.1)),
                },
                "radius": radius,
                "color": color,
            }
        )

    xyz.addSphere(
        {
            "center": {"x": center[0], "y": center[1], "z": center[2]},
            "opacity": 1,
            "radius": radius * 2,
            "color": "black",
        }
    )
    xyz.addSphere(
        {
            "center": {"x": center[0], "y": center[1], "z": -center[2]},
            "opacity": 1,
            "radius": radius * 2,
            "color": "black",
        }
    )
