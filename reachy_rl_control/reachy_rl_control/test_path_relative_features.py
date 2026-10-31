import math

import numpy as np

from path_relative_features import (
    build_v2_observation,
    compute_path_relative_features,
    path_length,
    resample_path,
)


ATOL = 1e-6


def assert_close(
    actual,
    expected,
    message,
):
    if not np.allclose(
        actual,
        expected,
        atol=ATOL,
        rtol=0.0,
    ):
        raise AssertionError(
            f"{message}\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )


def rotate_point(
    point,
    angle,
):
    x, y = point

    c = math.cos(angle)
    s = math.sin(angle)

    return (
        c * x - s * y,
        s * x + c * y,
    )


print(
    "\n===================================="
)
print(
    "PATH-RELATIVE V2 SELF TEST"
)
print(
    "====================================\n"
)


# -------------------------------------------------
# TEST 1: TRANSLATION INVARIANCE
# -------------------------------------------------

path_a = [
    (0.0, 0.0),
    (2.0, 0.0),
]

robot_a = (
    0.60,
    0.15,
    0.10,
)

features_a, _ = (
    compute_path_relative_features(
        robot_x=robot_a[0],
        robot_y=robot_a[1],
        robot_yaw=robot_a[2],
        path=path_a,
    )
)

shift_x = 4.0
shift_y = -3.0

path_b = [
    (
        x + shift_x,
        y + shift_y,
    )
    for x, y in path_a
]

robot_b = (
    robot_a[0] + shift_x,
    robot_a[1] + shift_y,
    robot_a[2],
)

features_b, _ = (
    compute_path_relative_features(
        robot_x=robot_b[0],
        robot_y=robot_b[1],
        robot_yaw=robot_b[2],
        path=path_b,
    )
)

assert_close(
    features_a,
    features_b,
    "Translation invariance FAILED.",
)

print(
    "PASS 1: translation invariance"
)


# -------------------------------------------------
# TEST 2: ROTATION INVARIANCE
# -------------------------------------------------

rotation = 1.20

path_rotated = [
    rotate_point(
        point,
        rotation,
    )
    for point in path_a
]

robot_xy_rotated = rotate_point(
    robot_a[:2],
    rotation,
)

features_rotated, _ = (
    compute_path_relative_features(
        robot_x=robot_xy_rotated[0],
        robot_y=robot_xy_rotated[1],
        robot_yaw=(
            robot_a[2]
            + rotation
        ),
        path=path_rotated,
    )
)

assert_close(
    features_a,
    features_rotated,
    "Rotation invariance FAILED.",
)

print(
    "PASS 2: rotation invariance"
)


# -------------------------------------------------
# TEST 3: 2 M / 10 M START EQUIVALENCE
# -------------------------------------------------

path_2m = [
    (0.0, 0.0),
    (2.0, 0.0),
]

path_10m = [
    (0.0, 0.0),
    (10.0, 0.0),
]

features_2m, meta_2m = (
    compute_path_relative_features(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        path=path_2m,
    )
)

features_10m, meta_10m = (
    compute_path_relative_features(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        path=path_10m,
    )
)

assert_close(
    features_2m,
    features_10m,
    "2 m / 10 m start equivalence FAILED.",
)

assert_close(
    features_2m,
    np.asarray(
        [
            0.25,
            0.0,
            0.80,
            0.0,
            0.0,
            0.0,
            2.0,
        ],
        dtype=np.float32,
    ),
    "Unexpected start-state features.",
)

print(
    "PASS 3: 2 m and 10 m starts "
    "produce the same path features"
)


# -------------------------------------------------
# TEST 4: L-SHAPE LOOK-AHEAD SEES TURN
# -------------------------------------------------

l_shape = [
    (0.0, 0.0),
    (2.0, 0.0),
    (2.0, 1.5),
]

features_l, meta_l = (
    compute_path_relative_features(
        robot_x=1.60,
        robot_y=0.0,
        robot_yaw=0.0,
        path=l_shape,
    )
)

next_x = float(
    features_l[0]
)

next_y = float(
    features_l[1]
)

lookahead_x = float(
    features_l[2]
)

lookahead_y = float(
    features_l[3]
)

if abs(next_y) > ATOL:
    raise AssertionError(
        "Immediate waypoint should still "
        "be on the straight section."
    )

if lookahead_y <= 0.0:
    raise AssertionError(
        "Look-ahead point did not reveal "
        "the upcoming left turn."
    )

print(
    "PASS 4: L-shape look-ahead "
    "reveals the corner"
)

print(
    f"        next local      = "
    f"({next_x:.3f}, {next_y:.3f})"
)

print(
    f"        look-ahead local = "
    f"({lookahead_x:.3f}, "
    f"{lookahead_y:.3f})"
)


# -------------------------------------------------
# TEST 5: PROGRESS WORKS ON L-SHAPE
# -------------------------------------------------

_, progress_meta = (
    compute_path_relative_features(
        robot_x=2.0,
        robot_y=0.75,
        robot_yaw=math.pi / 2.0,
        path=l_shape,
    )
)

expected_progress_distance = 2.75

expected_progress_ratio = (
    2.75 / 3.5
)

assert_close(
    progress_meta[
        "progress_distance"
    ],
    expected_progress_distance,
    "L-shape progress distance FAILED.",
)

assert_close(
    progress_meta[
        "progress_ratio"
    ],
    expected_progress_ratio,
    "L-shape progress ratio FAILED.",
)

print(
    "PASS 5: progress metric works "
    "through the L-shape"
)

print(
    "        progress distance = "
    f"{progress_meta['progress_distance']:.3f} m"
)

print(
    "        progress ratio    = "
    f"{100.0 * progress_meta['progress_ratio']:.1f}%"
)


# -------------------------------------------------
# TEST 6: UNIFORM PATH RESAMPLING
# -------------------------------------------------

resampled = resample_path(
    l_shape,
    spacing=0.25,
)

if resampled[0] != (0.0, 0.0):
    raise AssertionError(
        "Resampled path start changed."
    )

assert_close(
    resampled[-1],
    (2.0, 1.5),
    "Resampled path endpoint changed.",
)

print(
    "PASS 6: 0.25 m path resampling"
)

print(
    "        original length = "
    f"{path_length(l_shape):.3f} m"
)

print(
    "        resampled points = "
    f"{len(resampled)}"
)


# -------------------------------------------------
# TEST 7: COMPLETE OBSERVATION DIMENSION
# -------------------------------------------------

lidar = np.ones(
    12,
    dtype=np.float32,
)

observation, observation_meta = (
    build_v2_observation(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        vx=0.0,
        vy=0.0,
        wz=0.0,
        path=path_2m,
        lidar_normalized=lidar,
    )
)

if observation.shape != (22,):
    raise AssertionError(
        "Expected V2 observation "
        f"shape (22,), got "
        f"{observation.shape}."
    )

if observation.dtype != np.float32:
    raise AssertionError(
        "Expected float32 observation."
    )

print(
    "PASS 7: complete V2 observation "
    "has dimension 22 and dtype float32"
)


print(
    "\n===================================="
)
print(
    "ALL V2 PATH-RELATIVE TESTS PASSED"
)
print(
    "===================================="
)
