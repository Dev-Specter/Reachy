import math
from typing import Sequence, Tuple

import numpy as np


Point2D = Tuple[float, float]

DEFAULT_WAYPOINT_SPACING = 0.25
DEFAULT_LOOKAHEAD_DISTANCE = 0.80
DEFAULT_REMAINING_DISTANCE_CAP = 2.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def path_length(
    path: Sequence[Point2D],
) -> float:

    if len(path) < 2:
        return 0.0

    total = 0.0

    for start, end in zip(
        path[:-1],
        path[1:],
    ):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])

        total += math.hypot(dx, dy)

    return total


def point_at_path_distance(
    path: Sequence[Point2D],
    distance_along_path: float,
) -> Point2D:
    """
    Return the world-frame point lying at the requested
    arc length along a piecewise-linear path.
    """

    if len(path) == 0:
        raise ValueError(
            "Path must contain at least one point."
        )

    if len(path) == 1:
        return (
            float(path[0][0]),
            float(path[0][1]),
        )

    total = path_length(path)

    target = float(
        np.clip(
            distance_along_path,
            0.0,
            total,
        )
    )

    accumulated = 0.0

    for start, end in zip(
        path[:-1],
        path[1:],
    ):
        x0, y0 = map(float, start)
        x1, y1 = map(float, end)

        dx = x1 - x0
        dy = y1 - y0

        segment_length = math.hypot(
            dx,
            dy,
        )

        if segment_length <= 1e-12:
            continue

        if (
            accumulated + segment_length
            >= target
        ):
            fraction = (
                target - accumulated
            ) / segment_length

            return (
                x0 + fraction * dx,
                y0 + fraction * dy,
            )

        accumulated += segment_length

    return (
        float(path[-1][0]),
        float(path[-1][1]),
    )


def resample_path(
    path: Sequence[Point2D],
    spacing: float = DEFAULT_WAYPOINT_SPACING,
) -> list[Point2D]:
    """
    Resample a polyline at approximately uniform arc-length
    spacing while preserving the final endpoint.
    """

    if spacing <= 0.0:
        raise ValueError(
            "Spacing must be greater than zero."
        )

    total = path_length(path)

    if total <= 1e-12:
        return [
            (
                float(path[0][0]),
                float(path[0][1]),
            )
        ]

    distances = list(
        np.arange(
            0.0,
            total,
            spacing,
        )
    )

    if (
        not distances
        or abs(distances[-1] - total)
        > 1e-9
    ):
        distances.append(total)

    return [
        point_at_path_distance(
            path,
            distance,
        )
        for distance in distances
    ]


def project_onto_path(
    robot_x: float,
    robot_y: float,
    path: Sequence[Point2D],
):
    """
    Project the robot onto the closest segment of the path.

    Returns:
        projected_x
        projected_y
        progress_distance
        signed_cross_track_error
        path_heading
        segment_index
    """

    if len(path) < 2:
        raise ValueError(
            "Path must contain at least two points."
        )

    px = float(robot_x)
    py = float(robot_y)

    best = None
    accumulated = 0.0

    for segment_index, (
        start,
        end,
    ) in enumerate(
        zip(
            path[:-1],
            path[1:],
        )
    ):
        x0, y0 = map(float, start)
        x1, y1 = map(float, end)

        dx = x1 - x0
        dy = y1 - y0

        length_squared = (
            dx * dx
            + dy * dy
        )

        segment_length = math.sqrt(
            length_squared
        )

        if segment_length <= 1e-12:
            continue

        projection_fraction = (
            (px - x0) * dx
            + (py - y0) * dy
        ) / length_squared

        projection_fraction = float(
            np.clip(
                projection_fraction,
                0.0,
                1.0,
            )
        )

        projected_x = (
            x0
            + projection_fraction * dx
        )

        projected_y = (
            y0
            + projection_fraction * dy
        )

        error_x = px - projected_x
        error_y = py - projected_y

        distance_squared = (
            error_x * error_x
            + error_y * error_y
        )

        tangent_x = (
            dx / segment_length
        )

        tangent_y = (
            dy / segment_length
        )

        signed_cross_track_error = (
            tangent_x * error_y
            - tangent_y * error_x
        )

        progress_distance = (
            accumulated
            + projection_fraction
            * segment_length
        )

        path_heading = math.atan2(
            dy,
            dx,
        )

        candidate = (
            distance_squared,
            projected_x,
            projected_y,
            progress_distance,
            signed_cross_track_error,
            path_heading,
            segment_index,
        )

        if (
            best is None
            or candidate[0] < best[0]
        ):
            best = candidate

        accumulated += segment_length

    if best is None:
        raise RuntimeError(
            "Path contains no valid segments."
        )

    return best[1:]


def world_vector_to_robot_frame(
    dx_world: float,
    dy_world: float,
    robot_yaw: float,
) -> Point2D:
    """
    Rotate a world-frame vector into the robot frame.

    Robot-frame +x = forward.
    Robot-frame +y = left.
    """

    cosine = math.cos(robot_yaw)
    sine = math.sin(robot_yaw)

    local_x = (
        cosine * dx_world
        + sine * dy_world
    )

    local_y = (
        -sine * dx_world
        + cosine * dy_world
    )

    return (
        local_x,
        local_y,
    )


def compute_path_relative_features(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    path: Sequence[Point2D],
    waypoint_spacing: float = DEFAULT_WAYPOINT_SPACING,
    lookahead_distance: float = DEFAULT_LOOKAHEAD_DISTANCE,
    remaining_distance_cap: float = DEFAULT_REMAINING_DISTANCE_CAP,
):
    """
    Compute path-relative state.

    Feature order:
        0 next_waypoint_local_x
        1 next_waypoint_local_y
        2 lookahead_local_x
        3 lookahead_local_y
        4 signed_cross_track_error
        5 heading_error
        6 remaining_path_length_capped

    Also returns metadata containing true progress.
    """

    if waypoint_spacing <= 0.0:
        raise ValueError(
            "Waypoint spacing must be > 0."
        )

    if lookahead_distance <= 0.0:
        raise ValueError(
            "Lookahead distance must be > 0."
        )

    if remaining_distance_cap <= 0.0:
        raise ValueError(
            "Remaining-distance cap must be > 0."
        )

    (
        projected_x,
        projected_y,
        progress_distance,
        cross_track_error,
        path_heading,
        segment_index,
    ) = project_onto_path(
        robot_x,
        robot_y,
        path,
    )

    total_length = path_length(path)

    next_distance = min(
        progress_distance
        + waypoint_spacing,
        total_length,
    )

    lookahead_path_distance = min(
        progress_distance
        + lookahead_distance,
        total_length,
    )

    next_world = point_at_path_distance(
        path,
        next_distance,
    )

    lookahead_world = (
        point_at_path_distance(
            path,
            lookahead_path_distance,
        )
    )

    next_local = (
        world_vector_to_robot_frame(
            next_world[0] - robot_x,
            next_world[1] - robot_y,
            robot_yaw,
        )
    )

    lookahead_local = (
        world_vector_to_robot_frame(
            lookahead_world[0]
            - robot_x,
            lookahead_world[1]
            - robot_y,
            robot_yaw,
        )
    )

    heading_error = wrap_angle(
        path_heading
        - robot_yaw
    )

    remaining_distance = max(
        total_length
        - progress_distance,
        0.0,
    )

    remaining_capped = min(
        remaining_distance,
        remaining_distance_cap,
    )

    if total_length > 1e-12:
        progress_ratio = float(
            np.clip(
                progress_distance
                / total_length,
                0.0,
                1.0,
            )
        )
    else:
        progress_ratio = 1.0

    features = np.asarray(
        [
            next_local[0],
            next_local[1],
            lookahead_local[0],
            lookahead_local[1],
            cross_track_error,
            heading_error,
            remaining_capped,
        ],
        dtype=np.float32,
    )

    metadata = {
        "projected_x": projected_x,
        "projected_y": projected_y,
        "segment_index": segment_index,
        "progress_distance": progress_distance,
        "progress_ratio": progress_ratio,
        "remaining_distance": remaining_distance,
        "path_heading": path_heading,
        "next_waypoint_world": next_world,
        "lookahead_world": lookahead_world,
    }

    return features, metadata


def build_v2_observation(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    vx: float,
    vy: float,
    wz: float,
    path: Sequence[Point2D],
    lidar_normalized: Sequence[float],
):
    """
    Build the complete V2 human-aware observation.

    Observation order:
        vx
        vy
        wz
        next_waypoint_local_x
        next_waypoint_local_y
        lookahead_local_x
        lookahead_local_y
        cross_track_error
        heading_error
        remaining_path_length_capped
        lidar_sector_0 ... lidar_sector_11

    Total dimension = 22.
    """

    lidar = np.asarray(
        lidar_normalized,
        dtype=np.float32,
    )

    if lidar.shape != (12,):
        raise ValueError(
            "Expected exactly 12 LiDAR sectors, "
            f"received shape {lidar.shape}."
        )

    path_features, metadata = (
        compute_path_relative_features(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
            path=path,
        )
    )

    observation = np.concatenate(
        (
            np.asarray(
                [
                    vx,
                    vy,
                    wz,
                ],
                dtype=np.float32,
            ),
            path_features,
            lidar,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    if observation.shape != (22,):
        raise RuntimeError(
            "V2 observation should contain "
            f"22 values, got {observation.shape}."
        )

    return observation, metadata
