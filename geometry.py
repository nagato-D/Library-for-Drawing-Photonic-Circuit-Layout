"""
Description: Basic geometric components and operations.
Author: Di Yu
Contact: yudi.0211@foxmail.com
"""

import math
import numpy as np
from scipy.special import fresnel
from scipy.spatial.distance import euclidean

def rotate_array(point, origin, angle):
    """
    Apply the rotation operation to a two-column numpy array.

    Args:
        point (numpy.ndarray): A list of points to be rotated.
        origin (tuple): Position of the origin point.
        angle (float): Rotation angle in radian.
    
    Returns:
        numpy.ndarray: A list of rotated points.
    """
    ox, oy = origin
    px, py = point[:, 0], point[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.stack([qx, qy], 1)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Args:
        origin (tuple): Position of the origin point.
        point (tuple): Position of the point to be rotated.
        angle (float): Rotation angle in radian.

    Returns:
        tuple: Position of the rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy])

def scale_array(point, origin, scale):
    """
    Apply the scaling operation to a two-column numpy array.

    Args:
        point (numpy.ndarray): A list of points to be scaled.
        origin (tuple): Position of the origin point.
        scale (float): Scaling factor.
    
    Returns:
        numpy.ndarray: A list of scaled points.
    """
    ox, oy = origin
    px, py = point[:, 0], point[:, 1]

    qx = ox + scale * (px - ox)
    qy = oy + scale * (py - oy)
    return np.stack([qx, qy], 1)

def mirror(point, mirror_line):
    """
    Mirror a point over a line.

    Args:
        point (tuple): Position of the point to be mirrored.
        mirror_line (list): Mirror the bend over a line through points 1 and 2.

    Returns:
        tuple: Position of the mirrored point.
    """
    # Calculate the slope of the mirror line
    mirror_point1 = mirror_line[0]
    mirror_point2 = mirror_line[1]
    if mirror_point1[0] == mirror_point2[0]:
        # Handle the case when the mirror line is vertical
        slope = None
    else:
        slope = (mirror_point2[1] - mirror_point1[1]) / (mirror_point2[0] - mirror_point1[0])

    # Calculate the equation of the mirror line in the form y = mx + b
    if slope is not None:
        b = mirror_point1[1] - slope * mirror_point1[0]

    # Calculate the x-coordinate of the reflected point
    if slope is None:
        # Mirror line is vertical
        x_reflected = 2 * mirror_point1[0] - point[0]
        y_reflected = point[1]
    else:
        # Mirror line is not vertical
        b0 = point[1] - slope * point[0]
        distance_from_line = (b0 - b) / np.sqrt(slope**2 + 1) # Distance from point to mirror line
        normal_vector = np.array([slope, -1]) / np.sqrt(slope**2 + 1) # Normal vector of the mirror line
        x_reflected = 2 * distance_from_line * normal_vector[0] + point[0]
        y_reflected = 2 * distance_from_line * normal_vector[1] + point[1]
    return (x_reflected, y_reflected)

def circular_arc(bend_radius, bend_angle, number_of_points):
    """
    A circular arc with given radius and angle.

    Args:
        bend_radius (float): Radius of the circular arc in um.
        bend_angle (float): Angle of the circular arc in radian.
        number_of_points (int): Number of points used to approximate the circular arc.
    
    Returns:
        numpy.ndarray: A list of points that approximate the circular arc.
    """
    angles = np.linspace(0, bend_angle, number_of_points)
    x = bend_radius * np.cos(angles - np.pi/2)
    y = bend_radius * np.sin(angles - np.pi/2) + bend_radius
    if bend_angle > 0:
        return np.stack([x, y], 1)
    elif bend_angle < 0:
        return np.stack([x, -y], 1)
    else:
        raise ValueError("bend_angle must be nonzero")

def euler_curve(bend_radius, bend_angle, number_of_points):
    """
    An euler curve where the curvature changes linearly with the arc length.

    Args:
        bend_radius (float): Minimum curvature radius of the euler curve in um.
        bend_angle (float): Angle of the euler curve in radian.
        number_of_points (int): Number of points used to approximate the euler curve.
    
    Returns:
        numpy.ndarray: A list of points that approximate the euler curve.

    References:
        1. https://github.com/flaport/eulerbend/blob/master/eulerbend_gdsfactory.ipynb
        2. https://en.wikipedia.org/wiki/Euler_spiral
        3. https://en.wikipedia.org/wiki/Radius_of_curvature
    """
    L = bend_radius * abs(bend_angle)  # HALF of total length
    s = np.linspace(0, L, number_of_points // 2)
    f = np.sqrt(np.pi * bend_radius * L)
    if bend_angle > 0:
        y1, x1 = fresnel(s / f)
        x1, y1 = f * x1, f * y1
        # first, rotate by the final angle clockwise
        x2, y2 = np.dot(
            np.array([[np.cos(bend_angle), np.sin(bend_angle)], [-np.sin(bend_angle), np.cos(bend_angle)]]),
            np.stack([x1, y1], 0),
        )
        # then, flip along the x-axis (and reverse direction of the curve):
        x2, y2 = -x2[::-1], y2[::-1]
        # then translate from (x2[0], y2[0]) to (x1[-1], y1[-1])
        x2, y2 = x2 - x2[0] + x1[-1], y2 - y2[0] + y1[-1]
        x = np.concatenate([x1, x2], 0)
        y = np.concatenate([y1, y2], 0)
        return np.stack([x, y], 1)
    elif bend_angle < 0:
        y1, x1 = fresnel(s / f)
        # first, rotate by the final angle counterclockwise
        x2, y2 = np.dot(
            np.array([[np.cos(bend_angle), -np.sin(bend_angle)], [np.sin(bend_angle), np.cos(bend_angle)]]),
            np.stack([x1, y1], 0),
        )
        # then, flip along the x-axis (and reverse direction of the curve):
        x2, y2 = -x2[::-1], y2[::-1]
        # then translate from (x2[0], y2[0]) to (x1[-1], y1[-1])
        x2, y2 = x2 - x2[0] + x1[-1], y2 - y2[0] + y1[-1]
        x = f * np.concatenate([x1, x2], 0)
        # lastly, flip along the y-axis
        y = -f * np.concatenate([y1, y2], 0)
        return np.stack([x, y], 1)
    else:
        raise ValueError("bend_angle must be nonzero")

def euler_spiral(bend_radius, bend_angle, number_of_points):
    """
    An euler spiral where the radius of curvature changes linearly from infinity to the specified bend radius.

    Args:
        bend_radius (float): Minimum curvature radius of the euler spiral in um.
        bend_angle (float): Angle of the euler spiral in radian.
        number_of_points (int): Number of points used to approximate the euler spiral.
    
    Returns:
        numpy.ndarray: A list of points that approximate the euler spiral.
    """
    L = bend_radius * abs(bend_angle) * 2  # Length of the spiral
    s = np.linspace(0, L, number_of_points // 2)
    f = np.sqrt(np.pi * bend_radius * L)
    if bend_angle > 0:
        y, x = fresnel(s / f)
        x, y = f * x, f * y
        return np.stack([x, y], 1)

def size_euler_curve(bend_radius, bend_angle, number_of_points):
    """
    Calculate the size of an euler curve.

    Args:
        bend_radius (float): Minimum curvature radius of the euler curve in um.
        bend_angle (float): Angle of the euler curve in radian.
        number_of_points (int): Number of points used to approximate the euler curve.

    Returns:
        tuple: Width, height, and length of the euler curve.
    """
    curve = euler_curve(bend_radius, bend_angle, number_of_points)
    x_min = np.min(curve[:, 0])
    x_max = np.max(curve[:, 0])
    y_min = np.min(curve[:, 1])
    y_max = np.max(curve[:, 1])
    curve_length = 2 * bend_radius * bend_angle
    return (x_max-x_min, y_max-y_min, curve_length)

def euler_curve_S(bend_radius, bend_angle, number_of_points, straight_length=0):
    """
    Add an S-shaped euler curve.

    Args:
        bend_radius (float): Minimum curvature radius of the euler curve in um.
        bend_angle (float): Angle of the euler curve in radian.
        number_of_points (int): Number of points used to approximate the euler curve.
        straight_length (float): Length of the straight part in um, default to `0`.
    
    Returns:
        numpy.ndarray: A list of points that approximate the euler curve.
    
    Note:
        The length of this S-bend is 2 * bend_radius * abs(bend_angle).
    """
    L = bend_radius * abs(bend_angle)  # HALF of total length
    s = np.linspace(0, L, number_of_points // 2)
    f = np.sqrt(np.pi * bend_radius * L)
    if bend_angle > 0:
        y1, x1 = fresnel(s / f)
        x1, y1 = f * x1, f * y1
        # first, invert around the coordinate origin
        x2, y2 = -x1, -y1
        # then, reverse the order of points
        x2, y2 = np.flip(x2), np.flip(y2)
        # then translate from (x2[0], y2[0]) to (x1[-1], y1[-1])
        x2, y2 = x2 - x2[0] + x1[-1], y2 - y2[0] + y1[-1]
        # introduce the straight section
        x2, y2 = x2 + straight_length * np.cos(bend_angle/2), y2 + straight_length * np.sin(bend_angle/2)
        x = np.concatenate([x1, x2], 0)
        y = np.concatenate([y1, y2], 0)
        return np.stack([x, y], 1)
    elif bend_angle < 0:
        y1, x1 = fresnel(s / f)
        x1, y1 = f * x1, f * y1
        # first, invert around the coordinate origin
        x2, y2 = -x1, -y1
        # then, reverse the order of points
        x2, y2 = np.flip(x2), np.flip(y2)
        # then translate from (x2[0], y2[0]) to (x1[-1], y1[-1])
        x2, y2 = x2 - x2[0] + x1[-1], y2 - y2[0] + y1[-1]
        # introduce the straight section
        x2, y2 = x2 + straight_length * np.cos(bend_angle/2), y2 + straight_length * np.sin(bend_angle/2)
        x = np.concatenate([x1, x2], 0)
        y = -np.concatenate([y1, y2], 0)
        return np.stack([x, y], 1)
    else:
        raise ValueError("bend_angle must be nonzero")

def size_euler_curve_S(bend_radius, bend_angle, number_of_points, straight_length=0):
    """
    Calculate the size of an S-shaped euler curve.

    Args:
        bend_radius (float): Minimum curvature radius of the euler curve in um.
        bend_angle (float): Angle of the euler curve in radian.
        number_of_points (int): Number of points used to approximate the euler curve.
        straight_length (float): Length of the straight part in um, default to `0`.

    Returns:
        tuple: Width, height, and length of the euler curve.
    """
    curve = euler_curve_S(bend_radius, bend_angle, number_of_points, straight_length)
    x_min = np.min(curve[:, 0])
    x_max = np.max(curve[:, 0])
    y_min = np.min(curve[:, 1])
    y_max = np.max(curve[:, 1])
    curve_length = bend_radius * abs(bend_angle) * 2 + straight_length
    return (x_max-x_min, y_max-y_min, curve_length)

def euler_curve_S_connection(begin_point, end_point, bend_angle, direction, number_of_points):
    """
    A linearly transformed euler curve that connects two points.

    Args:
        begin_point (tuple): Position of the begin point.
        end_point (tuple): Position of the end point.
        bend_angle (float): Angle of the euler curve in radian.
        direction (str): Direction of the ending sections of euler curve, either 'x' or 'y'.
        number_of_points (int): Number of points used to approximate the euler curve.

    Returns:
        numpy.ndarray: A list of points that approximate the euler curve.
    
    Note:
        The created curve might not have a curvature radius that linearly changes with curve length.
    """
    bend_radius = 1
    curve = euler_curve_S(bend_radius, bend_angle, number_of_points, straight_length=0)
    if direction == 'x':
        None
    elif direction == 'y':
        curve = np.flip(curve, axis=1)
    else:
        raise ValueError("direction must be either 'x' or 'y'.")
    curve[:, 0] = curve[:, 0]*abs(end_point[0]-begin_point[0])/curve[-1, 0]
    curve[:, 1] = curve[:, 1]*abs(end_point[1]-begin_point[1])/curve[-1, 1]
    if end_point[1] > begin_point[1]:
        if end_point[0] > begin_point[0]:
            # End point is on the upper right of the begin point
            curve[:, 0] = curve[:, 0] + begin_point[0]
            curve[:, 1] = curve[:, 1] + begin_point[1]
        elif end_point[0] < begin_point[0]:
            # End point is on the uppder left of the begin point
            curve[:, 0] = -curve[:, 0]
            curve[:, 0] = curve[:, 0] + begin_point[0]
            curve[:, 1] = curve[:, 1] + begin_point[1]
        else:
            raise ValueError("begin_point and end_point must have different x-coordinates.")
    elif end_point[1] < begin_point[1]:
        if end_point[0] > begin_point[0]:
            # End point is on the lower right of the begin point
            curve[:, 1] = -curve[:, 1]
            curve[:, 0] = curve[:, 0] + begin_point[0]
            curve[:, 1] = curve[:, 1] + begin_point[1]
        elif end_point[0] < begin_point[0]:
            # End point is on the lower left of the begin point
            curve[:, 0] = -curve[:, 0]
            curve[:, 1] = -curve[:, 1]
            curve[:, 0] = curve[:, 0] + begin_point[0]
            curve[:, 1] = curve[:, 1] + begin_point[1]
        else:
            raise ValueError("begin_point and end_point must have different y-coordinates.")
    return curve

def euler_curve_protrusion(straight_length, bend_radius, bend_orientation, number_of_points):
    """
    A protrusion curve made of four sections of euler spirals and one straight part.

    Args:
        straight_length (float): Length of the straight part in um.
        bend_radius (float): Minimum curvature radius of the euler bends in um.
        bend_orientation (float): Change in the orientation angle across the euler waveguide bend coupler in radian.
        number_of_points (int): Number of points used to approximate the euler curve.
    
    Returns:
        numpy.ndarray: A list of points that approximate the euler-spiral protrusion.
    """
    # Sections of the waveguide coupler
    protrusion_section1 = euler_curve(bend_radius, bend_orientation, number_of_points)
    protrusion_section2 = -1 * np.flip(protrusion_section1, axis=0) + protrusion_section1[-1] * 2
    protrusion_section3 = np.array([
        protrusion_section2[-1], 
        [straight_length+protrusion_section2[-1, 0], protrusion_section2[-1, 1]]
    ])
    protrusion_section4 = np.flip(protrusion_section2, axis=0) * np.array([-1, 1])
    protrusion_section4 += np.array([protrusion_section2[-1, 0] + protrusion_section3[-1, 0], 0])
    protrusion_section5 = np.flip(protrusion_section1, axis=0) * np.array([-1, 1])
    protrusion_section5 += np.array([protrusion_section1[-1, 0] + protrusion_section4[-1, 0], 0])

    # Assemble the sections of waveguide coupler
    protrusion_curve = np.concatenate(
        (protrusion_section1, protrusion_section2, protrusion_section3, protrusion_section4, protrusion_section5), 
        axis=0
    )

    return protrusion_curve

def circular_arc_S(bend_radius, bend_angle, number_of_points, straight_length=0):
    """
    Add an S-shaped circular arc.

    Args:
        bend_radius (float): Radius of the circular arc in um.
        bend_angle (float): Angle of the euler curve in radian.
        number_of_points (int): Number of points used to approximate the euler curve.
        straight_length (float): Length of the straight part in um, default to `0`.
    
    Returns:
        numpy.ndarray: A list of points that approximate the euler curve.
    
    Note:
        The length of this S-bend is 2 * bend_radius * abs(bend_angle).
    """
    theta = np.linspace(0, bend_angle, number_of_points)
    if bend_angle > 0:
        y1, x1 = (1 - np.cos(theta)) * bend_radius, np.sin(theta) * bend_radius
        # first, invert around the coordinate origin
        x2, y2 = -x1, -y1
        # then, reverse the order of points
        x2, y2 = np.flip(x2), np.flip(y2)
        # then translate from (x2[0], y2[0]) to (x1[-1], y1[-1])
        x2, y2 = x2 - x2[0] + x1[-1], y2 - y2[0] + y1[-1]
        # introduce the straight section
        x2, y2 = x2 + straight_length * np.cos(bend_angle/2), y2 + straight_length * np.sin(bend_angle/2)
        x = np.concatenate([x1, x2], 0)
        y = np.concatenate([y1, y2], 0)
        return np.stack([x, y], 1)
    elif bend_angle < 0:
        y1, x1 = -(1 - np.cos(theta)) * bend_radius, np.sin(-theta) * bend_radius
        # first, invert around the coordinate origin
        x2, y2 = -x1, -y1
        # then, reverse the order of points
        x2, y2 = np.flip(x2), np.flip(y2)
        # then translate from (x2[0], y2[0]) to (x1[-1], y1[-1])
        x2, y2 = x2 - x2[0] + x1[-1], y2 - y2[0] + y1[-1]
        # introduce the straight section
        x2, y2 = x2 + straight_length * np.cos(bend_angle/2), y2 + straight_length * np.sin(bend_angle/2)
        x = np.concatenate([x1, x2], 0)
        y = np.concatenate([y1, y2], 0)
        return np.stack([x, y], 1)
    else:
        raise ValueError("bend_angle must be nonzero")

def curve_length(points):
    """
    Calculate the length of a curve.

    Args:
        points (numpy.ndarray): A list of points that approximate the curve.

    Returns:
        float: Length of the curve.
    """
    length = 0
    for i in range(1, len(points)):
        length += euclidean(points[i-1], points[i])
    return length

def sine_curve(bend_radius, adiabaticity, number_of_points):
    """
    A sine curve with given radius and adiabaticity.

    Args:
        bend_radius (float): Radius of the sine curve in um.
        adiabaticity (float): Adiabaticity of the sine curve.
        number_of_points (int): Number of points used to approximate the sine curve.
    
    Returns:
        numpy.ndarray: A list of points that approximate the sine curve.
    """
    t = np.linspace(0, 1, number_of_points)
    t = t ** (1/2) # To make the curve more adiabatic
    x = np.pi / 2 * t * bend_radius * np.sqrt(adiabaticity)
    y = adiabaticity * bend_radius * np.sin(np.pi / 2 * t)
    return np.stack([x, y], 1)

def sine_curve_updated(bend_radius, adiabaticity, number_of_points, bending_direction, flip=False):
    """
    A sine curve with given radius and adiabaticity.
    
    Args:
        bend_radius (float): Radius of the sine curve in um.
        adiabaticity (float): Adiabaticity of the sine curve.
        number_of_points (int): Number of points used to approximate the sine curve.
        bending_direction (str): Direction of the sine curve, either 'up' or 'down'.
        flip (bool): Whether to flip the sine curve, default to `False`.
        
    Returns:
        numpy.ndarray: A list of points that approximate the sine curve.
    """
    t = np.linspace(0, 1, number_of_points)
    t = t ** (1/3) # To make the curve more adiabatic
    x = np.pi / 2 * t * bend_radius * np.sqrt(adiabaticity)
    y = adiabaticity * bend_radius * np.sin(np.pi / 2 * t)
    if flip:
        sine_curve = np.stack([x, y], 1)
        sine_curve += -sine_curve[-1]
        sine_curve = -sine_curve
        sine_curve = np.flip(sine_curve, axis=0)
        if bending_direction == 'up':
            return sine_curve
        elif bending_direction == 'down':
            return sine_curve * np.array([1, -1])
    else:
        orientation = np.arctan(np.sqrt(adiabaticity))
        sine_curve = rotate_array(np.stack([x, y], 1), [0, 0], -orientation)
        if bending_direction == 'up':
            return sine_curve * np.array([1, -1])
        elif bending_direction == 'down':
            return sine_curve

def archimedean_spiral(a, b, angular_range, number_of_points):
    """
    An Archimedean spiral with given parameters.

    Args:
        a (float): The initial distance of the spiral from the origin.
        b (float): Increase in the distance from the origin per radian.
        angular_range (tuple): The angular range of the spiral in radian.
        number_of_points (int): Number of points used to approximate the spiral.
    
    Returns:
        numpy.ndarray: A list of points that approximate the Archimedean spiral.
    """
    t = np.linspace(angular_range[0], angular_range[1], number_of_points)
    x = (a + b * t) * np.cos(t)
    y = (a + b * t) * np.sin(t)
    x = x - x[0]
    y = y - y[0]
    return np.stack([x, y], 1)