"""
Description: Basic integrated photonic components.
Author: Di Yu
Contact: yudi.0211@foxmail.com
Last update: 2024-08-18
"""

import gdspy
import geometry
import numpy as np

def waveguide(cell, wg_length, wg_width, etch_width, angle, center, layer_wg, layer_wg_etch, use_begin_point=False):
    """
    Add straight waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide will be added.
        length (float): Length of the waveguide in um.
        width (float): Width of the waveguide in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide in um.
        angle (float): Tilt angle of the waveguide in radian.
        center (tuple): Position of the waveguide in um.
        layer_wg (int): Layer index of the waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide.
        use_begin_point (bool): Use the beginning point coordinate for the `center` argument if `True`, default to `False`.

    Returns:
        gdspy.cell: A gdspy cell with the waveguide.
    """
    if use_begin_point:
        begin_point = center
        wg_core = gdspy.Rectangle(
            (begin_point[0], -wg_width/2+begin_point[1]), 
            (wg_length+begin_point[0], wg_width/2+begin_point[1]), 
            layer=layer_wg
        ) # Waveguide core
        cell.add(wg_core.rotate(angle, begin_point))

        wg_etch = gdspy.Rectangle(
            (begin_point[0], -wg_width/2-etch_width+begin_point[1]),
            (wg_length+begin_point[0], wg_width/2+etch_width+begin_point[1]),
            layer=layer_wg_etch
        ) # Waveguide cladding
        cell.add(wg_etch.rotate(angle, begin_point))

    else:
        wg_core = gdspy.Rectangle(
            (-wg_length/2+center[0], -wg_width/2+center[1]), 
            (wg_length/2+center[0], wg_width/2+center[1]), 
            layer=layer_wg
        ) # Waveguide core
        cell.add(wg_core.rotate(angle, center))

        wg_etch = gdspy.Rectangle(
            (-wg_length/2+center[0], -wg_width/2-etch_width+center[1]), 
            (wg_length/2+center[0], wg_width/2+etch_width+center[1]), 
            layer=layer_wg_etch
        ) # Waveguide cladding
        cell.add(wg_etch.rotate(angle, center))

    return cell

def waveguide_bend_circular(
        cell, 
        bend_width, 
        etch_width, 
        bend_radius, 
        bend_angle, 
        bend_orientation, 
        begin_point=(0, 0), 
        tolerance=1e-3, 
        layer_bend=1, 
        layer_wg_etch=2
    ):
    """
    Add a circular arc waveguide bend.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide bend will be added.
        bend_width (float): Width of the waveguide bend in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide bend in um.
        bend_radius (float): Radius of the circular waveguide bend in um.
        bend_angle (float): Angle rangle of the circular waveguide bend in radian.
        bend_orientation (float): Orientation of the input port of the bend in radian.
        begin_point (tuple): Position of the beginning point of the waveguide bend in um, default to `(0, 0)`.
        mirror_line (list): Mirror the bend over a line through points 1 and 2, default to `None`.
        tolerance (float): Approximate curvature resolution, default to `1e-3`.
        layer_bend (int): Layer index of the waveguide bend, default to `1`.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide bend, default to `2`.

    Returns:
        gdspy.cell: A gdspy cell with the waveguide bend.
        tuple: Position of the end point of the waveguide bend.
    """
    bend_center = begin_point + bend_radius * np.array([np.cos(bend_orientation+np.pi/2), np.sin(bend_orientation+np.pi/2)])
    bend_core = gdspy.Round(
        center=bend_center, 
        radius=bend_radius+bend_width/2, 
        inner_radius=bend_radius-bend_width/2, 
        initial_angle=bend_orientation-np.pi/2, 
        final_angle=bend_orientation+bend_angle-np.pi/2, 
        tolerance=tolerance, 
        max_points=0, 
        layer=layer_bend
    ) # Waveguide bend core

    etch_bend = gdspy.Round(
        center=bend_center, 
        radius=bend_radius+bend_width/2+etch_width, 
        inner_radius=bend_radius-bend_width/2-etch_width, 
        initial_angle=bend_orientation-np.pi/2, 
        final_angle=bend_orientation+bend_angle-np.pi/2, 
        tolerance=tolerance, 
        max_points=0, 
        layer=layer_wg_etch
    ) # Waveguide bend cladding

    end_point = geometry.rotate(bend_center, begin_point, bend_angle)

    cell.add(etch_bend)
    cell.add(bend_core)
    return cell, end_point

def waveguide_bend_sine(
        cell,
        bend_width,
        etch_width,
        adiabaticity,
        bend_radius,
        bend_orientation,
        begin_point=(0, 0),
        mirror_line=None,
        number_of_points=64,
        inverse=False,
        layer_bend=1,
        layer_wg_etch=2,
        bend_length_correction=0,
        end_type=('flush', 'flush')
    ):
    """
    Add a sine-shaped waveguide bend.
    
    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide bend will be added.
        bend_width (float): Width of the waveguide bend in um.
        etch_width (float): Width of the etching regions in each side of the waveguide bend in um.
        adiabaticity (float): A parameter that determines the length of the waveguide bend.
        bend_radius (float): Radius of the waveguide bend in um.
        bend_orientation (float): Orientation of the input port of the bend in radian.
        begin_point (tuple): Position of the beginning point of the waveguide bend in um, default to `(0, 0)`.
        mirror_line (str): Mirror the bend over x-axis or y-axis if `'x'` or `'y'`, default to `None`.
        number_of_points (int): Number of points used to approximate the waveguide bend, default to `64`.
        inverse (bool): Inverse the waveguide bend and exchange the position of begin and end points if `True`, default to `False`.
        layer_bend (int): Layer index of the waveguide bend, default to `1`.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide bend, default to `2`.
        bend_length_correction (float): Correction of the length of the waveguide bend, default to `0.05`.
        end_type (tuple): Type of end caps for the paths. A 2-element tuple represents the start and end extensions to the paths. The element could be `'flush'` or `'round'`.
    
    Returns:
        gdspy.cell: A gdspy cell with the waveguide bend.
        tuple: Position of the end point of the waveguide bend.
    """
    begin_point = np.array(begin_point)

    if not inverse:
        bend_center_curve = geometry.sine_curve(bend_radius, adiabaticity, number_of_points)
        if mirror_line == 'x':
            bend_center_curve[:, 1] = -bend_center_curve[:, 1]
        elif mirror_line == 'y':
            bend_center_curve[:, 0] = -bend_center_curve[:, 0]
        bend_center_curve += begin_point # Translation
        bend_core = gdspy.FlexPath(bend_center_curve, bend_width, layer=layer_bend, ends=end_type)
        etch_bend = gdspy.FlexPath(bend_center_curve, bend_width+2*etch_width, layer=layer_wg_etch, ends=end_type)

        bend_core = bend_core.rotate(bend_orientation, begin_point) # Rotation
        etch_bend = etch_bend.rotate(bend_orientation, begin_point)
        end_point = geometry.rotate(begin_point, bend_center_curve[-1], bend_orientation)
    else:
        bend_center_curve = geometry.sine_curve(bend_radius, adiabaticity, number_of_points)
        if mirror_line == 'x':
            bend_center_curve[:, 1] = -bend_center_curve[:, 1]
        elif mirror_line == 'y':
            bend_center_curve[:, 0] = -bend_center_curve[:, 0]
        bend_center_curve = bend_center_curve - bend_center_curve[-1] # Translation end point to origin
        bend_center_curve = -bend_center_curve # Inverse the curve with respect to the origin
        bend_center_curve = np.flip(bend_center_curve, 0) # Reverse the curve
        bend_center_curve += begin_point

        begin_point_2D = np.reshape(begin_point, (-1, 2))

        # Correct beginning orientation
        begin_direction_vector = np.array([np.cos(bend_orientation), np.sin(bend_orientation)])
        bend_center_curve = np.concatenate([begin_point_2D, bend_center_curve+bend_length_correction/2*begin_direction_vector], 0)

        # Correct terminating orientation
        bend_angle = np.arctan(np.sqrt(adiabaticity))
        end_direction_vector = np.array([np.cos(bend_orientation+bend_angle), np.sin(bend_orientation+bend_angle)])
        bend_center_curve = np.concatenate([bend_center_curve, bend_center_curve[-1:]+bend_length_correction/2*end_direction_vector], 0)

        bend_core = gdspy.FlexPath(bend_center_curve, bend_width, layer=layer_bend, ends=end_type).rotate(bend_orientation, begin_point_2D)
        etch_bend = gdspy.FlexPath(bend_center_curve, bend_width+2*etch_width, layer=layer_wg_etch, ends=end_type).rotate(bend_orientation, begin_point)
        end_point = geometry.rotate(begin_point, bend_center_curve[-1], bend_orientation)

    cell.add(etch_bend)
    cell.add(bend_core)
    return cell, end_point

def waveguide_bend_euler(
        cell, 
        bend_width, 
        etch_width, 
        bend_radius, 
        bend_angle, 
        bend_orientation, 
        begin_point=(0, 0), 
        mirror_line=None, 
        number_of_points=64, 
        layer_bend=1, 
        layer_wg_etch=2
    ):
    """
    Add an euler-type waveguide bend connecting two waveguides with different orientation.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide bend will be added.
        bend_width (float): Width of the waveguide bend in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide bend in um.
        bend_radius (float): Minimum curvature radius of the euler bends in um.
        bend_angle (float): Change in orientation angle accross the euler waveguide bend in radian.
        bend_orientation (float): Initial orientation angle of the euler waveguide bend in radian.
        begin_point (tuple): Position of the beginning point of the waveguide bend in um, default to `(0, 0)`.
        mirror_line (list): Mirror the bend over a line through points 1 and 2, default to `None`.
        number_of_points (int): Number of points used to approximate the euler waveguide bend, default to `64`.
        layer_bend (int): Layer index of the waveguide bend, default to `1`.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide bend, default to `2`.

    Returns:
        gdspy.cell: A gdspy cell with the waveguide bend.
        tuple: Position of the end point of the waveguide bend.
    """
    bend_center_curve = geometry.euler_curve(bend_radius, bend_angle, number_of_points)
    bend_center_curve += begin_point # Translation
    bend_core = gdspy.FlexPath(bend_center_curve, bend_width, layer=layer_bend)
    etch_bend = gdspy.FlexPath(bend_center_curve, bend_width+2*etch_width, layer=layer_wg_etch)

    if mirror_line is None:
        bend_core = bend_core.rotate(bend_orientation, begin_point) # Rotation
        etch_bend = etch_bend.rotate(bend_orientation, begin_point)
        end_point = geometry.rotate(begin_point, bend_center_curve[-1], bend_orientation)
    else:
        bend_core = bend_core.mirror(mirror_line).rotate(bend_orientation, begin_point) # Rotation and mirror reflection
        etch_bend = etch_bend.mirror(mirror_line).rotate(bend_orientation, begin_point)
        end_point = geometry.rotate(begin_point, bend_center_curve[-1], bend_orientation)
        end_point = geometry.mirror(end_point, mirror_line)

    cell.add(etch_bend)
    cell.add(bend_core)
    return cell, end_point

def waveguide_bend_sine_extend(
        cell,
        bend_width,
        etch_width,
        bend_radius,
        bend_angle,
        adiabaticity,
        bend_orientation,
        begin_point=(0, 0),
        number_of_points=64,
        layer_bend=1,
        layer_wg_etch=2,
        bend_length_correction=0
    ):
    """
    Add a waveguide bend that connects two straight waveguides with a continuously changing curvature.
    A adiabatic sine-shaped bend is incorporated to bridge the I/O straight waveguides and the circular waveguide bend.
    Waveguide bends created with this function are more compact than the regular euler bends.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide bend will be added.
        bend_width (float): Width of the waveguide bend in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide bend in um.
        bend_radius (float): Curvature radius of the circular arc bend in um.
        bend_angle (float): Change in orientation angle accross the waveguide bend in radian.
        adiabaticity (float): A parameter that determines the length of the sine adiabatic bend section, ranging from 0 to 1.
        bend_orientation (float): Initial orientation angle of the waveguide bend in radian.
        begin_point (tuple): Position of the beginning point of the waveguide bend in um, default to `(0, 0)`.
        number_of_points (int): Number of points used to approximate the waveguide bend, default to `64`.
        layer_bend (int): Layer index of the waveguide bend, default to `1`.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide bend, default to `2`.

    Returns:
        gdspy.cell: A gdspy cell with the waveguide bend.
        tuple: Position of the end point of the waveguide bend.
    """

    # Add adiabatic sine-shaped bend
    if bend_angle > 0:
        bend_direction = 'up'
    else:
        bend_direction = 'down'
    
    sine_curve_1st = geometry.sine_curve_updated(
        bend_radius, 
        adiabaticity, 
        number_of_points, 
        bending_direction=bend_direction,
        flip=False
    )
    current_orientation = np.arctan(np.sqrt(adiabaticity)) * np.sign(bend_angle)

    # Add circular arc bend
    bend_angle_circular = bend_angle - current_orientation * 2
    circular_arc = geometry.circular_arc(bend_radius, bend_angle_circular, number_of_points)
    circular_arc = circular_arc * np.array([1, np.sign(bend_angle)]) # Flip y-coordinate
    circular_arc += sine_curve_1st[-1] # Translation
    if bend_angle > 0:
        circular_arc = geometry.rotate_array(circular_arc, circular_arc[0], current_orientation)
    if bend_angle < 0:
        circular_arc = geometry.rotate_array(circular_arc, circular_arc[0], np.pi + current_orientation)
    current_orientation += bend_angle_circular

    # Add adiabatic sine-shaped bend
    sine_curve_2nd = geometry.sine_curve_updated(
        bend_radius, 
        adiabaticity, 
        number_of_points, 
        bending_direction=bend_direction, 
        flip=True
    )
    sine_curve_2nd += circular_arc[-1] # Translation
    sine_curve_2nd = geometry.rotate_array(sine_curve_2nd, sine_curve_2nd[0], current_orientation)
    current_orientation += np.arctan(np.sqrt(adiabaticity)) * np.sign(bend_angle)

    # Define waveguide bend structure
    bend_center_curve = np.concatenate([sine_curve_1st, circular_arc[1:, :], sine_curve_2nd[1:, :]], 0)

    # Correct the beginning orientation
    bend_center_curve = np.concatenate(
        [np.zeros([1, 2]), bend_center_curve + bend_length_correction/2*np.array([1, 0])], 0
    )

    # Correct the terminating orientation
    bend_center_curve = np.concatenate(
        [bend_center_curve, bend_center_curve[-1:] + \
         bend_length_correction/2*np.array([np.cos(bend_angle), np.sin(bend_angle)])], 0
    )
    bend_center_curve += begin_point # Translation
    bend_core = gdspy.FlexPath(bend_center_curve, bend_width, layer=layer_bend).rotate(bend_orientation, begin_point)
    etch_bend = gdspy.FlexPath(bend_center_curve, bend_width+2*etch_width, layer=layer_wg_etch).rotate(bend_orientation, begin_point)
    end_point = geometry.rotate(begin_point, bend_center_curve[-1], bend_orientation)

    cell.add(etch_bend)
    cell.add(bend_core)
    return cell, end_point

def waveguide_bend_euler_S(
        cell,
        bend_width,
        etch_width,
        bend_radius,
        bend_angle,
        bend_orientation,
        begin_point=(0, 0),
        mirror_line=None,
        number_of_points=64,
        layer_bend=1,
        layer_wg_etch=2, 
        straight_length=0
    ):
    """
    Add an euler-type waveguide bend connecting two waveguides with different y-coordinates.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide bend will be added.
        bend_width (float): Width of the waveguide bend in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide bend in um.
        bend_radius (float): Minimum curvature radius of the euler bends in um.
        bend_angle (float): Change in orientation angle accross the euler waveguide bend in radian.
        bend_orientation (float): Initial orientation angle of the euler waveguide bend in radian.
        begin_point (tuple): Position of the beginning point of the waveguide bend in um, default to `(0, 0)`.
        mirror_line (list): Mirror the bend over a line through points 1 and 2, default to `None`.
        number_of_points (int): Number of points used to approximate the euler waveguide bend, default to `64`.
        layer_bend (int): Layer index of the waveguide bend, default to `1`.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide bend, default to `2`.
        straight_length (float): Length of the straight waveguide part in um, default to `0`.
    
    Returns:
        gdspy.cell: A gdspy cell with the waveguide bend.
        tuple: Position of the end point of the waveguide bend.
    """
    bend_center_curve = geometry.euler_curve_S(bend_radius, bend_angle, number_of_points, straight_length)
    bend_center_curve += begin_point # Translation
    bend_core = gdspy.FlexPath(bend_center_curve, bend_width, layer=layer_bend)
    etch_bend = gdspy.FlexPath(bend_center_curve, bend_width+2*etch_width, layer=layer_wg_etch)

    if mirror_line is None:
        bend_core = bend_core.rotate(bend_orientation, begin_point) # Rotation
        etch_bend = etch_bend.rotate(bend_orientation, begin_point)
        end_point = geometry.rotate(begin_point, bend_center_curve[-1], bend_orientation)
    else:
        bend_core = bend_core.mirror(mirror_line).rotate(bend_orientation, begin_point) # Rotation and mirror reflection
        etch_bend = etch_bend.mirror(mirror_line).rotate(bend_orientation, begin_point)
        end_point = geometry.rotate(begin_point, bend_center_curve[-1], bend_orientation)
        end_point = geometry.mirror(end_point, mirror_line)

    cell.add(etch_bend)
    cell.add(bend_core)
    return cell, end_point

def waveguide_bend_circular_S(
        cell,
        bend_width,
        etch_width,
        begin_point,
        end_point,
        number_of_points=128,
        layer_bend=1,
        layer_wg_etch=2, 
    ):
    """
    Add a circular arc waveguide bend connecting two waveguides with different y-coordinates.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide bend will be added.
        bend_width (float): Width of the waveguide bend in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide bend in um.
        begin_point (tuple): Position of the beginning point of the waveguide bend in um.
        end_point (tuple): Position of the end point of the waveguide bend in um.
        number_of_points (int): Number of points used to approximate the euler waveguide bend, default to `64`.
        layer_bend (int): Layer index of the waveguide bend, default to `1`.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide bend, default to `2`.
    
    Returns:
        gdspy.cell: A gdspy cell with the waveguide bend.
    """
    # Calculate radius and angle of circular arc
    x_span = end_point[0] - begin_point[0]
    y_span = end_point[1] - begin_point[1]
    if y_span != 0:
        arc_radius = (x_span**2 + y_span**2) / (4*abs(y_span))
        arc_angle = np.arcsin(abs(x_span)/2/arc_radius)*np.sign(y_span)
        bend_center_curve = geometry.circular_arc_S(arc_radius, arc_angle, number_of_points)
        bend_center_curve += begin_point # Translation
    else:
        bend_center_curve = np.stack([begin_point, end_point], 0)
    bend_core = gdspy.FlexPath(bend_center_curve, bend_width, layer=layer_bend)
    etch_bend = gdspy.FlexPath(bend_center_curve, bend_width+2*etch_width, layer=layer_wg_etch)

    cell.add(etch_bend)
    cell.add(bend_core)
    return cell, end_point

def pulley_coupler(
        cell, 
        wg_width, 
        etch_width, 
        coupling_length, 
        bend_radius, 
        bend_orientation, 
        angle, 
        center, 
        number_of_points, 
        layer_wg, 
        layer_wg_etch, 
        return_size_only=False
    ):
    """
    Add a waveguide bend coupler.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide bend coupler will be added.
        wg_width (float): Width of the waveguide bend coupler in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide bend coupler in um.
        coupling_length (float): Length of the coupling region in um.
        bend_radius (float): Minimum curvature radius of the euler bends in um.
        bend_orientation (float): Change in the orientation angle across the euler waveguide bend coupler in radian.
        angle (float): Tilt angle of the waveguide bend coupler in radian.
        center (tuple): Position of the waveguide bend coupler in um.
        number_of_points (int): Number of points used to approximate one half of the euler waveguide bend.
        layer_wg (int): Layer index of the waveguide bend coupler.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide bend coupler.
        return_size_only (bool): Only return the size of the waveguide bend coupler if `True`, default to `False`.

    Returns:
        gdspy.cell: A gdspy cell with the waveguide bend coupler.
        tuple: Size of the waveguide bend coupler.
    """
    coupler_curve = geometry.euler_curve_protrusion(coupling_length, bend_radius, bend_orientation, number_of_points)
    coupler_size = [max(np.abs(coupler_curve[:, 0])), max(np.abs(coupler_curve[:, 1]))]
    if return_size_only:
        return coupler_size
    coupler_curve += np.array(list(center)) - coupler_curve[-1] / 2 # Translation
    coupler_core = gdspy.FlexPath(coupler_curve, wg_width, layer=layer_wg).to_polygonset()
    coupler_clad = gdspy.FlexPath(coupler_curve, wg_width+2*etch_width, layer=layer_wg_etch).to_polygonset()
    cell.add(coupler_core.rotate(angle, center)) # Rotation
    cell.add(coupler_clad.rotate(angle, center))
    return cell, coupler_size

def grating(
        cell, 
        wg_length, 
        wg_width, 
        etch_width, 
        post_num, 
        post_radius, 
        post_gap, 
        post_period, 
        center, 
        layer_wg, 
        layer_wg_etch, 
        layer_grating, 
        wg_length_left=0, 
        wg_length_right=0, 
        post_arrange='one-side', 
        grating_offset_x=0
    ):
    """
    Add a post-array grating nearby a straight waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the grating will be added.
        wg_length (float): Length of the straight waveguide in um.
        wg_width (float): Width of the straight waveguide in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide/ring in um.
        post_num (int): Number of posts that form the grating.
        post_radius (float): Radius of the posts in um.
        post_gap (float): Separation distance between the ring and the post array in um.
        post_period (float): Period of the post array in um.
        center (tuple): Position of the grating microring in um.
        layer_wg (int): Layer index of the straight waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide.
        layer_grating (int): Layer index of the grating.
        wg_length_left (float): Length of the straight waveguide on the left side in um, default to `0`.
        wg_length_right (float): Length of the straight waveguide on the right right in um, default to `0`.
        post_arrange (str): 'one-side'/'two-side' for posts in one/both sides of the waveguide, default to `'one-side'`.
        grating_offset_x (float): Offset of the grating in x-direction in um, default to `0`.

    Returns:
        gdspy.cell: A gdspy cell with the grating microring and waveguide.
    """
    # Add straight bus waveguide on the left of the grating
    if wg_length_left == 0:
        wg_length_left = wg_length/2
    center_left = (center[0] - wg_length_left/2, center[1])
    cell = waveguide(cell, wg_length_left, wg_width, etch_width, 0, center_left, layer_wg, layer_wg_etch)

    # Add straight bus waveguide on the right of the grating
    if wg_length_right == 0:
        wg_length_right = wg_length/2
    center_right = (center[0] + wg_length_right/2, center[1])
    cell = waveguide(cell, wg_length_right, wg_width, etch_width, 0, center_right, layer_wg, layer_wg_etch)

    # Add periodic posts (grating)
    grating_length = post_num * post_period # length of grating
    if post_arrange == 'one-side':
        for idx_post in range(post_num):
            post_x = center[0] - grating_length/2 + idx_post*post_period + grating_offset_x
            post_y = center[1] + wg_width/2 + post_gap + post_radius
            post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
            cell.add(post)

    if post_arrange == 'two-side':
        for idx_post in range(post_num):
            post_x = center[0] - grating_length/2 + idx_post*post_period + grating_offset_x
            post_y = center[1] + wg_width/2 + post_gap + post_radius
            post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
            cell.add(post)
            post_y = center[1] - wg_width/2 - post_gap - post_radius
            post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
            cell.add(post)

    return cell

def grating_zigzag(
        cell, 
        wg_length, 
        wg_width, 
        etch_width, 
        period_num, 
        sawtooth_width, 
        sawtooth_height, 
        grating_width, 
        grating_gap, 
        grating_period, 
        center, 
        layer_wg, 
        layer_wg_etch, 
        layer_grating, 
        wg_length_left=0, 
        wg_length_right=0, 
        grating_arrange='one-side', 
        grating_offset_x=0
    ):
    """
    Add a post-array grating nearby a straight waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the grating will be added.
        wg_length (float): Length of the straight waveguide in um.
        wg_width (float): Width of the straight waveguide in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide/ring in um.
        period_num (int): Number of periods that form the grating.
        sawtooth_width (float): Width of the sawtooth in um, measured at the narrowest position.
        sawtooth_height (float): Height of the sawtooth in um.
        grating_width (float): Width of the grating in um, memasured at the narrowest position.
        grating_gap (float): Separation distance between the ring and the grating in um.
        grating_period (float): Period of the grating in um.
        center (tuple): Position of the grating microring in um.
        layer_wg (int): Layer index of the straight waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide.
        layer_grating (int): Layer index of the grating.
        wg_length_left (float): Length of the straight waveguide on the left side in um, default to `0`.
        wg_length_right (float): Length of the straight waveguide on the right right in um, default to `0`.
        grating_arrange (str): 'one-side'/'two-side' for grating in one/both sides of the waveguide, default to `'one-side'`.
        grating_offset_x (float): Offset of the grating in x-direction in um, default to `0`.

    Returns:
        gdspy.cell: A gdspy cell with the grating microring and waveguide.
    """
    # Add straight bus waveguide on the left of the provided center position
    if wg_length_left == 0:
        wg_length_left = wg_length/2
    center_left = (center[0] - wg_length_left/2, center[1])
    cell = waveguide(cell, wg_length_left, wg_width, etch_width, 0, center_left, layer_wg, layer_wg_etch)

    # Add straight bus waveguide on the right of the provided center position
    if wg_length_right == 0:
        wg_length_right = wg_length/2
    center_right = (center[0] + wg_length_right/2, center[1])
    cell = waveguide(cell, wg_length_right, wg_width, etch_width, 0, center_right, layer_wg, layer_wg_etch)

    # Add zigzag grating
    grating_length = period_num * grating_period # length of grating
    if grating_arrange == 'one-side':
        for idx_period in range(period_num):
            period_pts = [
                (sawtooth_width/2, 0),
                (grating_period/2-sawtooth_width/2, sawtooth_height),
                (grating_period/2, sawtooth_height),
                (grating_period/2, sawtooth_height+grating_width),
                (-grating_period/2, sawtooth_height+grating_width),
                (-grating_period/2, sawtooth_height),
                (-grating_period/2+sawtooth_width/2, sawtooth_height),
                (-sawtooth_width/2, 0)
            ]
            single_sawtooth = gdspy.Polygon(period_pts, layer=layer_grating)
            period_x = center[0] - grating_length/2 + idx_period*grating_period + grating_offset_x
            period_y = center[1] + wg_width/2 + grating_gap
            cell.add(single_sawtooth.translate(period_x, period_y))

    if grating_arrange == 'two-side':
        for idx_period in range(period_num):
            period_pts = [
                (sawtooth_width/2, 0),
                (grating_period/2-sawtooth_width/2, sawtooth_height),
                (grating_period/2, sawtooth_height),
                (grating_period/2, sawtooth_height+grating_width),
                (-grating_period/2, sawtooth_height+grating_width),
                (-grating_period/2, sawtooth_height),
                (-grating_period/2+sawtooth_width/2, sawtooth_height),
                (-sawtooth_width/2, 0)
            ]
            single_sawtooth = gdspy.Polygon(period_pts, layer=layer_grating)
            period_x = center[0] - grating_length/2 + idx_period*grating_period + grating_offset_x
            period_y = center[1] + wg_width/2 + grating_gap
            cell.add(single_sawtooth.translate(period_x, period_y))
            period_y = center[1] - wg_width/2 - grating_gap
            cell.add(single_sawtooth.mirror((1, 0)).translate(period_x, period_y))
    return cell

def ring(cell, ring_radius, ring_width, etch_width, center, layer_ring, layer_wg_etch):
    """
    Add ring resonator.
    
    Args:
        cell (gdspy.cell): A gdspy cell in which the ring resonator will be added.
        ring_radius (float): Radius of the ring in um.
        ring_width (float): Width of the ring in um.
        etch_width (float): Width of the exposure regions in both sides of the ring in um.
        center (tuple): Position of the ring in um.
        layer_ring (int): Layer index of the ring core.
        layer_wg_etch (int): Layer index of the etch regions around the ring.

    Returns:
        gdspy.cell: A gdspy cell with the ring resonator.
    """
    # Add ring resonator
    ring = gdspy.Round(
        center=center, 
        radius=ring_radius + ring_width/2, 
        inner_radius=ring_radius - ring_width/2, 
        layer=layer_ring
    ) # Ring core
    cell.add(ring)
    etch_ring = gdspy.Round(
        center=center, 
        radius=ring_radius + ring_width/2 + etch_width, 
        inner_radius=ring_radius - ring_width/2 - etch_width, 
        layer=layer_wg_etch
    ) # Ring cladding
    cell.add(etch_ring)
    return cell

def loaded_microring(
        cell, 
        wg_length, 
        wg_width, 
        ring_radius, 
        ring_width, 
        gap, 
        etch_width, 
        center, 
        layer_wg, 
        layer_wg_etch
    ):
    """
    Add a ring resonator coupled to a straight waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the grating will be added.
        wg_length (float): Length of the straight waveguide in um.
        wg_width (float): Width of the straight waveguide in um.
        ring_radius (float): Radius of the ring in um.
        ring_width (float): Width of the ring in um.
        gap (float): Separation distance between the ring and the straight waveguide in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide/ring in um.
        center (tuple): Position of the grating microring in um.
        layer_wg (int): Layer index of the grating microring.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide/ring.

    Returns:
        gdspy.cell: A gdspy cell with the grating microring and waveguide.
    """
    # Add straight bus waveguide
    wg = gdspy.Rectangle(
        (-wg_length/2+center[0], -wg_width/2+center[1]), 
        (wg_length/2+center[0], wg_width/2+center[1]), 
        layer=layer_wg
    ) # Waveguide core
    cell.add(wg)
    etch_wg = gdspy.Rectangle(
        (-wg_length/2+center[0], -wg_width/2-etch_width+center[1]), 
        (wg_length/2+center[0], wg_width/2+etch_width+center[1]), 
        layer=layer_wg_etch
    ) # Waveguide cladding
    cell.add(etch_wg)

    # Add ring resonator
    ring_center = (center[0], wg_width/2 + gap + ring_width/2 + ring_radius + +center[1])
    ring = gdspy.Round(
        center=ring_center, 
        radius=ring_radius + ring_width/2, 
        inner_radius=ring_radius - ring_width/2, 
        layer=layer_wg
    ) # Ring core
    cell.add(ring)
    etch_ring = gdspy.Round(
        center=ring_center, 
        radius=ring_radius + ring_width/2 + etch_width, 
        inner_radius=ring_radius - ring_width/2 - etch_width, 
        layer=layer_wg_etch
    ) # Ring cladding
    cell.add(etch_ring)
    return cell

def racetrack(
        cell, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        etch_width, 
        center, 
        number_of_points, 
        layer_racetrack, 
        layer_wg_etch, 
        protrusion=(0, 0)
    ):
    """
    Add racetrack resonator with euler-spipral waveguide bends.

    Args:
        cell (gdspy.cell): A gdspy cell in which the racetrack resonator will be added.
        racetrack_length (float): Length of the racetrack in um.
        racetrack_radius (float): Minimum curvature radius of the racetrack in um.
        racetrack_width (float): Width of the racetrack in um.
        etch_width (float): Width of the exposure regions in both sides of the racetrack in um.
        center (tuple): Position of the racetrack in um.
        number_of_points (int): Number of points used to approximate the racetrack bend.
        layer_racetrack (int): Layer index of the racetrack.
        layer_wg_etch (int): Layer index of the etch regions around the racetrack.
        protrusion (tuple): Width and height of the rectangular protrusion in um, default to `(0, 0)`.

    Returns:
        gdspy.cell: A gdspy cell with the racetrack resonator.

    Warning:
        This function uses a wrong length expression of the euler bend and needs to be updated.
    """
    euler_bend_length = 2 * np.pi * racetrack_radius # Length of the euler bend
    straight_length = racetrack_length / 2 - euler_bend_length # Length of the straight waveguide
    euler_bend = geometry.euler_curve(racetrack_radius, np.pi, number_of_points) # Euler bend in the racetrack
    racetrack_height =  euler_bend[-1][1] # Height of the racetrack
    left_bend_begin_point = (center[0]-straight_length/2, center[1]+racetrack_height/2)
    right_bend_begin_point = (center[0]+straight_length/2, center[1]-racetrack_height/2)

    # Add straight waveguide at the top
    top_wg_center = (left_bend_begin_point[0] + straight_length / 2, left_bend_begin_point[1])
    cell = waveguide(
        cell, 
        straight_length, 
        racetrack_width, 
        etch_width, 
        0, 
        top_wg_center, 
        layer_racetrack, 
        layer_wg_etch
    )

    # Center of waveguide at the bottom
    bottom_wg_center = (right_bend_begin_point[0] - straight_length / 2, right_bend_begin_point[1])
    if protrusion == (0, 0):
        # Add straight waveguide at the bottom
        cell = waveguide(
            cell, 
            straight_length, 
            racetrack_width, 
            etch_width, 
            np.pi, 
            bottom_wg_center, 
            layer_racetrack, 
            layer_wg_etch
        )
    else:
        # Add waveguide protrusion at the bottom
        protrusion_curve = [
            (left_bend_begin_point[0], right_bend_begin_point[1]), 
            (bottom_wg_center[0]-protrusion[0]/2, right_bend_begin_point[1]), 
            (bottom_wg_center[0]-protrusion[0]/2, right_bend_begin_point[1]+protrusion[1]), 
            (bottom_wg_center[0]+protrusion[0]/2, right_bend_begin_point[1]+protrusion[1]), 
            (bottom_wg_center[0]+protrusion[0]/2, right_bend_begin_point[1]), 
            (right_bend_begin_point[0], right_bend_begin_point[1])
        ]
        protrusion_core = gdspy.FlexPath(protrusion_curve, racetrack_width, layer=layer_racetrack)
        etch_protrusion = gdspy.FlexPath(protrusion_curve, racetrack_width+2*etch_width, layer=layer_wg_etch)
        cell.add(protrusion_core)
        cell.add(etch_protrusion)

    # Add euler bend on the left side
    cell, bend_end_point = waveguide_bend_euler(
        cell, 
        racetrack_width, 
        etch_width, 
        racetrack_radius, 
        bend_angle=np.pi, 
        bend_orientation=np.pi, 
        begin_point=left_bend_begin_point, 
        mirror_line=None, 
        number_of_points=128, 
        layer_bend=layer_racetrack, 
        layer_wg_etch=layer_wg_etch
    )

    # Add euler bend on the right side
    cell, bend_end_point = waveguide_bend_euler(
        cell, 
        racetrack_width, 
        etch_width, 
        racetrack_radius, 
        bend_angle=np.pi, 
        bend_orientation=0, 
        begin_point=right_bend_begin_point, 
        mirror_line=None, 
        number_of_points=128, 
        layer_bend=layer_racetrack, 
        layer_wg_etch=layer_wg_etch
    )

    return cell

def racetrack_circular(
        cell, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        etch_width, 
        center, 
        number_of_points, 
        layer_racetrack, 
        layer_wg_etch
    ):
    """
    Add racetrack resonator with semicircular waveguide bends.

    Args:
        cell (gdspy.cell): A gdspy cell in which the racetrack resonator will be added.
        racetrack_length (float): Length of the racetrack in um.
        racetrack_radius (float): Radius of the waveguide bend in um.
        racetrack_width (float): Width of the racetrack in um.
        etch_width (float): Width of the exposure regions in both sides of the racetrack in um.
        center (tuple): Position of the racetrack in um.
        number_of_points (int): Number of points used to approximate the racetrack bend.
        layer_racetrack (int): Layer index of the racetrack.
        layer_wg_etch (int): Layer index of the etch regions around the racetrack.

    Returns:
        gdspy.cell: A gdspy cell with the racetrack resonator.

    Warning:
        This function uses a wrong length expression of the euler bend and needs to be updated.
    """
    semicircular_bend_length = np.pi * racetrack_radius # Length of the semicircular bend
    straight_length = racetrack_length / 2 - semicircular_bend_length # Length of the straight waveguide
    semicircular_bend = geometry.circular_arc(racetrack_radius, np.pi, number_of_points) # Semicircular bend in the racetrack
    racetrack_height =  semicircular_bend[-1][1] # Height of the racetrack
    left_bend_begin_point = (center[0]-straight_length/2, center[1]+racetrack_height/2)
    right_bend_begin_point = (center[0]+straight_length/2, center[1]-racetrack_height/2)

    # Add straight waveguide at the top
    top_wg_center = (left_bend_begin_point[0] + straight_length / 2, left_bend_begin_point[1])
    cell = waveguide(
        cell, 
        straight_length, 
        racetrack_width, 
        etch_width, 
        0, 
        top_wg_center, 
        layer_racetrack, 
        layer_wg_etch
    )

    # Add straight waveguide at the bottom
    bottom_wg_center = (right_bend_begin_point[0] - straight_length / 2, right_bend_begin_point[1])
    cell = waveguide(
        cell, 
        straight_length, 
        racetrack_width, 
        etch_width, 
        np.pi, 
        bottom_wg_center, 
        layer_racetrack, 
        layer_wg_etch
    )

    # Add euler bend on the left side
    cell, bend_end_point = waveguide_bend_circular(
        cell, 
        racetrack_width, 
        etch_width, 
        racetrack_radius, 
        bend_angle=np.pi, 
        bend_orientation=np.pi, 
        begin_point=left_bend_begin_point, 
        mirror_line=None, 
        number_of_points=128, 
        layer_bend=layer_racetrack, 
        layer_wg_etch=layer_wg_etch
    )

    # Add euler bend on the right side
    cell, bend_end_point = waveguide_bend_circular(
        cell, 
        racetrack_width, 
        etch_width, 
        racetrack_radius, 
        bend_angle=np.pi, 
        bend_orientation=0, 
        begin_point=right_bend_begin_point, 
        mirror_line=None, 
        number_of_points=128, 
        layer_bend=layer_racetrack, 
        layer_wg_etch=layer_wg_etch
    )

    return cell

def loaded_racetrack(
        cell, 
        wg_length, 
        wg_width, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        gap, 
        coupling_length, 
        bend_orientation, 
        bend_radius, 
        etch_width, 
        center, 
        number_of_points, 
        layer_wg, 
        layer_wg_etch, 
        wg_length_left=0, 
        wg_length_right=0, 
        bend_type='euler'
    ):
    """
    Add a racetrack resonator coupled to a straight waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the racetrack resonator will be added.
        wg_length (float): Length of the bus waveguide in um.
        wg_width (float): Width of the bus waveguide in um.
        racetrack_length (float): Length of the racetrack in um.
        racetrack_radius (float): Minimum curvature radius of the racetrack in um.
        racetrack_width (float): Width of the racetrack in um.
        gap (float): Separation distance between the racetrack and the bus waveguide in um.
        coupling_length (float): Length of the coupling region in um.
        bend_orientation (float): Change in the orientation angle across the euler waveguide bend coupler in radian.
        bend_radius (float): Minimum curvature radius of the euler waveguide bend coupler in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide/racetrack in um.
        center (tuple): Position of the racetrack resonator in um.
        number_of_points (int): Number of points used to approximate the racetrack bend.
        layer_wg (int): Layer index of the racetrack and the bus waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the racetrack and the bus waveguide.
        wg_length_left (float): Length of the bus waveguide on the left side in um, default to `0`.
        wg_length_right (float): Length of the bus waveguide on the right right in um, default to `0`.
        bend_type (str): 'euler'/'circular' for euler-spiral/circular bend section, default to `'euler'`.
    
    Returns:
        gdspy.cell: A gdspy cell with the racetrack resonator and waveguide.
    
    Note:
        The `wg_length_left` and `wg_length_right` override the `wg_length` if they are not `None`.
        The x-span of the loaded racetrack is the sum of `coupler_size[0]`, `wg_length_left` and `wg_length_right`.
    """
    # Add waveguide coupler
    cell, coupler_size = pulley_coupler(
        cell, 
        wg_width, 
        etch_width, 
        coupling_length, 
        bend_radius, 
        bend_orientation, 
        0, 
        center, 
        number_of_points, 
        layer_wg, 
        layer_wg_etch
    )
    
    # Add bus waveguide on the left of the coupler
    if wg_length_left == 0:
        wg_length_left = wg_length/2 # Length of the bus waveguide on the left of the coupler
    bus_wg_x = center[0] - coupler_size[0]/2 - (wg_length_left-coupler_size[0]/2)/2 # x-coordinate of the bus waveguide center
    bus_wg_y = center[1] # y-coordinate of the bus waveguide center
    cell = waveguide(
        cell,
        wg_length_left-coupler_size[0]/2,
        wg_width,
        etch_width,
        0,
        (bus_wg_x, bus_wg_y),
        layer_wg,
        layer_wg_etch
    )

    # Add bus waveguide on the right of the coupler
    if wg_length_right == 0:
        wg_length_right = wg_length/2 # Length of the bus waveguide on the right of the coupler
    bus_wg_x = center[0] + coupler_size[0]/2 + (wg_length_right-coupler_size[0]/2)/2 # x-coordinate of the bus waveguide center
    bus_wg_y = center[1] # y-coordinate of the bus waveguide center
    cell = waveguide(
        cell,
        wg_length_right-coupler_size[0]/2,
        wg_width,
        etch_width,
        0,
        (bus_wg_x, bus_wg_y),
        layer_wg,
        layer_wg_etch
    )

    # Define the position of the racetrack
    if bend_type == 'euler':
        euler_bend = geometry.euler_curve(racetrack_radius, np.pi, number_of_points) # Euler bend in the racetrack
        racetrack_height = euler_bend[-1][1] # Height of the racetrack
        racetrack_center = (center[0], center[1] + wg_width/2 + gap + racetrack_width/2 + racetrack_height/2 + coupler_size[1])

        # Add racetrack resonator
        cell = racetrack(
            cell, 
            racetrack_length, 
            racetrack_radius, 
            racetrack_width, 
            etch_width, 
            racetrack_center, 
            number_of_points, 
            layer_wg, 
            layer_wg_etch
        )
    elif bend_type == 'circular':
        semicircular_bend = geometry.circular_arc(racetrack_radius, np.pi, number_of_points) # Semicircular bend in the racetrack
        racetrack_height = semicircular_bend[-1][1] # Height of the racetrack
        racetrack_center = (center[0], center[1] + wg_width/2 + gap + racetrack_width/2 + racetrack_height/2 + coupler_size[1])

        # Add racetrack resonator
        cell = racetrack_circular(
            cell, 
            racetrack_length, 
            racetrack_radius, 
            racetrack_width, 
            etch_width, 
            racetrack_center, 
            number_of_points, 
            layer_wg, 
            layer_wg_etch
        )
    else:
        raise ValueError('`bend_type` must be `euler` or `circular`.')
    
    return cell

def loaded_grating_racetrack(
        cell, 
        wg_length, 
        wg_width, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        gap, 
        coupling_length, 
        bend_orientation, 
        bend_radius, 
        etch_width, 
        post_num, 
        post_radius, 
        post_gap, 
        post_period, 
        center, 
        number_of_points, 
        layer_wg, 
        layer_wg_etch, 
        layer_grating, 
        wg_length_left=0, 
        wg_length_right=0, 
        post_arrange='one-side', 
        bend_type='euler'
    ):
    """
    Add a racetrack resonator with a post-array grating, coupled to a straight bus waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the grating will be added.
        wg_length (float): Length of the bus waveguide in um.
        wg_width (float): Width of the bus waveguide in um.
        racetrack_length (float): Length of the racetrack in um.
        racetrack_radius (float): Minimum curvature radius of the racetrack in um.
        racetrack_width (float): Width of the racetrack in um.
        gap (float): Separation distance between the racetrack and the bus waveguide in um.
        coupling_length (float): Length of the coupling region in um.
        bend_orientation (float): Change in the orientation angle across the euler waveguide bend coupler in radian.
        bend_radius (float): Minimum curvature radius of the euler waveguide bend coupler in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide/racetrack in um.
        post_num (int): Number of posts that form the grating.
        post_radius (float): Radius of the posts in um.
        post_gap (float): Separation distance between the racetrack and the post array in um.
        post_period (float): Period of the post array in um.
        center (tuple): Position of the grating racetrack in um.
        number_of_points (int): Number of points used to approximate the racetrack bend.
        layer_wg (int): Layer index of the bus waveguide and the racetrack.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide/racetrack.
        layer_grating (int): Layer index of the grating.
        wg_length_left (float): Length of the bus waveguide on the left side in um, default to `0`.
        wg_length_right (float): Length of the bus waveguide on the right right in um, default to `0`.
        post_arrange (str): 'one-side'/'two-side' for posts in one/both sides of the waveguide, default to `'one-side'`.
        bend_type (str): 'euler'/'circular' for euler-spiral/circular bend section, default to `'euler'`.

    Returns:
        gdspy.cell: A gdspy cell with the grating racetrack and waveguide.
    """
    # Add a bus waveguide and a racetrack resonator
    cell = loaded_racetrack(
        cell, 
        wg_length, 
        wg_width, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        gap, 
        coupling_length, 
        bend_orientation, 
        bend_radius, 
        etch_width, 
        center, 
        number_of_points, 
        layer_wg, 
        layer_wg_etch, 
        wg_length_left, 
        wg_length_right, 
        bend_type
    )

    # Determine size of the racetrack resonator
    if bend_type == 'euler':
        racetrack_euler_bend = geometry.euler_curve(racetrack_radius, np.pi, number_of_points) # Euler bend in the racetrack
        racetrack_height = racetrack_euler_bend[-1][1] # Height of the racetrack
        racetrack_straight_length = racetrack_length / 2 - np.pi * racetrack_radius * 2 # Length of the straight section in the racetrack
    elif bend_type == 'circular':
        racetrack_semicircular_bend = geometry.circular_arc(racetrack_radius, np.pi, number_of_points) # Semicircular bend in the racetrack
        racetrack_height = racetrack_semicircular_bend[-1][1] # Height of the racetrack
        racetrack_straight_length = racetrack_length / 2 - np.pi * racetrack_radius # Length of the straight section in the racetrack

    # Determine size of the coupler
    coupler_curve = geometry.euler_curve_protrusion(coupling_length, bend_radius, bend_orientation, number_of_points)
    coupler_height = max(coupler_curve[:, 1]) # Height of the coupler

    # Determine size of the grating
    grating_length = post_num * post_period
    if grating_length > racetrack_straight_length:
        raise ValueError('`grating_length` must be smaller than the length of the straight section in the racetrack.')

    # Add the grating formed by an array of posts
    if post_arrange == 'one-side':
        for idx_post in range(post_num):
            post_relative_pos = -grating_length/2 + idx_post*post_period # Relative position of the post to the top section of the racetrack
            if abs(post_relative_pos) <= racetrack_straight_length/2: # Post in the straight section
                post_x = center[0] + post_relative_pos
                post_y = center[1] + wg_width/2 + gap + racetrack_height - post_gap - post_radius + coupler_height
            
            if post_relative_pos > racetrack_straight_length/2: # Post in the right circular bend
                post_angle = (post_relative_pos - racetrack_straight_length/2) / racetrack_radius
                post_x = center[0] + racetrack_straight_length/2 + racetrack_radius * np.sin(post_angle)
                racetrack_center_y = center[1] + wg_width/2 + coupler_height + gap + racetrack_width/2 + racetrack_height/2
                post_y = racetrack_center_y + (racetrack_height/2 - racetrack_width/2 - post_gap - post_radius) * np.cos(post_angle)
            
            if post_relative_pos < -racetrack_straight_length/2: # Post in the left circular bend
                post_angle = -(post_relative_pos + racetrack_straight_length/2) / racetrack_radius
                post_x = center[0] - racetrack_straight_length/2 - racetrack_radius * np.sin(post_angle)
                racetrack_center_y = center[1] + wg_width/2 + coupler_height + gap + racetrack_width/2 + racetrack_height/2
                post_y = racetrack_center_y + (racetrack_height/2 - racetrack_width/2 - post_gap - post_radius) * np.cos(post_angle)
            post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
            cell.add(post)

    if post_arrange == 'two-side':
        for idx_post in range(post_num):
            post_relative_pos = -grating_length/2 + idx_post*post_period # Relative position of the post to the top section of the racetrack
            if abs(post_relative_pos) <= racetrack_straight_length/2: # Post in the straight section
                post_x = center[0] - grating_length/2 + idx_post*post_period
                post_y = center[1] + wg_width/2 + gap + racetrack_height - post_gap - post_radius + coupler_height
                post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
                cell.add(post) # Post in the lower side
                post_y = center[1] + wg_width/2 + gap + racetrack_height + racetrack_width + post_gap + post_radius + coupler_height
                post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
                cell.add(post) # Post in the upper side
            
            if post_relative_pos > racetrack_straight_length/2: # Post in the right circular bend
                post_angle = (post_relative_pos - racetrack_straight_length/2) / racetrack_radius
                # Radius of the post-array track in the inner side
                post_inner_radius = racetrack_height/2 - racetrack_width/2 - post_gap - post_radius
                # Radius of the post-array track in the outer side
                post_outer_radius = racetrack_height/2 + racetrack_width/2 + post_gap + post_radius
                # Add post in the inner side
                post_x = center[0] + racetrack_straight_length/2 + post_inner_radius * np.sin(post_angle)
                racetrack_center_y = center[1] + wg_width/2 + coupler_height + gap + racetrack_width/2 + racetrack_height/2
                post_y = racetrack_center_y + post_inner_radius * np.cos(post_angle)
                post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
                cell.add(post)
                # Add post in the outer side
                post_x = center[0] + racetrack_straight_length/2 + post_outer_radius * np.sin(post_angle)
                post_y = racetrack_center_y + post_outer_radius * np.cos(post_angle)
                post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
                cell.add(post) # Post in the outer side
                
            if post_relative_pos < -racetrack_straight_length/2: # Post in the left circular bend
                post_angle = -(post_relative_pos + racetrack_straight_length/2) / racetrack_radius
                # Radius of the post-array track in the inner side
                post_inner_radius = racetrack_height/2 - racetrack_width/2 - post_gap - post_radius
                # Radius of the post-array track in the outer side
                post_outer_radius = racetrack_height/2 + racetrack_width/2 + post_gap + post_radius
                # Add post in the inner side
                post_x = center[0] - racetrack_straight_length/2 - post_inner_radius * np.sin(post_angle)
                racetrack_center_y = center[1] + wg_width/2 + coupler_height + gap + racetrack_width/2 + racetrack_height/2
                post_y = racetrack_center_y + post_inner_radius * np.cos(post_angle)
                post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
                cell.add(post)
                # Add post in the outer side
                post_x = center[0] - racetrack_straight_length/2 - post_outer_radius * np.sin(post_angle)
                post_y = racetrack_center_y + post_outer_radius * np.cos(post_angle)
                post = gdspy.Round(center=(post_x, post_y), radius=post_radius, number_of_points=64, layer=layer_grating)
                cell.add(post) # Post in the outer side
    return cell

def loaded_zigzag_grating_racetrack(
        cell, 
        wg_length, 
        wg_width, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        gap, 
        coupling_length, 
        bend_orientation, 
        bend_radius, 
        etch_width, 
        period_num, 
        sawtooth_width, 
        sawtooth_height, 
        grating_width, 
        grating_gap, 
        grating_period, 
        center, 
        number_of_points, 
        layer_wg, 
        layer_wg_etch, 
        layer_grating, 
        wg_length_left=0, 
        wg_length_right=0, 
        grating_arrange='one-side', 
        bend_type='euler'
    ):
    """
    Add a racetrack resonator with a zigzag grating, coupled to a straight bus waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the grating will be added.
        wg_length (float): Length of the bus waveguide in um.
        wg_width (float): Width of the bus waveguide in um.
        racetrack_length (float): Length of the racetrack in um.
        racetrack_radius (float): Minimum curvature radius of the racetrack in um.
        racetrack_width (float): Width of the racetrack in um.
        gap (float): Separation distance between the racetrack and the bus waveguide in um.
        coupling_length (float): Length of the coupling region in um.
        bend_orientation (float): Change in the orientation angle across the euler waveguide bend coupler in radian.
        bend_radius (float): Minimum curvature radius of the euler waveguide bend coupler in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide/racetrack in um.
        period_num (int): Number of zigzag periods that form the grating.
        sawtooth_width (float): Width of the sawtooth in um, measured at the narrowest position.
        sawtooth_height (float): Height of the sawtooth in um.
        grating_width (float): Width of the grating in um, measured at the narrowest position.
        grating_gap (float): Separation distance between the racetrack and the grating in um.
        grating_period (float): Period of the grating in um.
        center (tuple): Position of the grating racetrack in um.
        number_of_points (int): Number of points used to approximate the racetrack bend.
        layer_wg (int): Layer index of the bus waveguide and the racetrack.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide/racetrack.
        layer_grating (int): Layer index of the grating.
        wg_length_left (float): Length of the bus waveguide on the left side in um, default to `0`.
        wg_length_right (float): Length of the bus waveguide on the right right in um, default to `0`.
        grating_arrange (str): 'one-side'/'two-side' for grating in one/both sides of the waveguide, default to `'one-side'`.
        bend_type (str): 'euler'/'circular' for euler-spiral/circular bend section, default to `'euler'`.

    Returns:
        gdspy.cell: A gdspy cell with the grating racetrack and waveguide.
    """
    # Add a bus waveguide and a racetrack resonator
    cell = loaded_racetrack(
        cell, 
        wg_length, 
        wg_width, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        gap, 
        coupling_length, 
        bend_orientation, 
        bend_radius, 
        etch_width, 
        center, 
        number_of_points, 
        layer_wg, 
        layer_wg_etch, 
        wg_length_left, 
        wg_length_right, 
        bend_type
    )

    # Determine size of the racetrack resonator
    if bend_type == 'euler':
        racetrack_euler_bend = geometry.euler_curve(racetrack_radius, np.pi, number_of_points) # Euler bend in the racetrack
        racetrack_height = racetrack_euler_bend[-1][1] # Height of the racetrack
        racetrack_straight_length = racetrack_length / 2 - np.pi * racetrack_radius * 2 # Length of the straight section in the racetrack
    elif bend_type == 'circular':
        racetrack_semicircular_bend = geometry.circular_arc(racetrack_radius, np.pi, number_of_points) # Semicircular bend in the racetrack
        racetrack_height = racetrack_semicircular_bend[-1][1] # Height of the racetrack
        racetrack_straight_length = racetrack_length / 2 - np.pi * racetrack_radius # Length of the straight section in the racetrack

    # Determine size of the coupler
    coupler_curve = geometry.euler_curve_protrusion(coupling_length, bend_radius, bend_orientation, number_of_points)
    coupler_height = max(coupler_curve[:, 1]) # Height of the coupler

    # Determine size of the grating
    grating_length = period_num * grating_period
    if grating_length > racetrack_straight_length:
        raise ValueError('`grating_length` must be smaller than the length of the straight section in the racetrack.')

    # Add the grating formed by an array of sawteeth
    period_pts = [
        (sawtooth_width/2, 0),
        (grating_period/2-sawtooth_width/2, sawtooth_height),
        (grating_period/2, sawtooth_height),
        (grating_period/2, sawtooth_height+grating_width),
        (-grating_period/2, sawtooth_height+grating_width),
        (-grating_period/2, sawtooth_height),
        (-grating_period/2+sawtooth_width/2, sawtooth_height),
        (-sawtooth_width/2, 0)
    ] # Points of a single sawtooth in the grating
    if grating_arrange == 'one-side':
        for idx_period in range(period_num):
            # Relative position of the sawtooth bottom side to the top section of the racetrack
            period_relative_pos = -grating_length/2 + idx_period*grating_period
            period_x = center[0] + period_relative_pos
            period_y = center[1] + wg_width/2 + gap + racetrack_height + racetrack_width + grating_gap + coupler_height
            single_sawtooth = gdspy.Polygon(period_pts, layer=layer_grating)
            cell.add(single_sawtooth.translate(period_x, period_y))
    if grating_arrange == 'two-side':
        for idx_period in range(period_num):
            # Relative position of the sawtooth bottom side to the top section of the racetrack
            period_relative_pos = -grating_length/2 + idx_period*grating_period
            period_x = center[0] - grating_length/2 + idx_period*grating_period
            period_y = center[1] + wg_width/2 + gap + racetrack_height + racetrack_width + grating_gap + coupler_height
            single_sawtooth = gdspy.Polygon(period_pts, layer=layer_grating)
            cell.add(single_sawtooth.translate(period_x, period_y)) # Grating in the upper side
            period_y = center[1] + wg_width/2 + gap + racetrack_height - grating_gap + coupler_height
            single_sawtooth = gdspy.Polygon(period_pts, layer=layer_grating)
            cell.add(single_sawtooth.mirror((1, 0).translate(period_x, period_y))) # Grating in the lower side
    return cell

def grating_microring(
        cell, 
        ring_radius, 
        ring_width, 
        etch_width, 
        post_period, 
        post_num, 
        post_gap, 
        post_radius, 
        post_angle_offset,
        ring_center, 
        tolerance, 
        layer_wg, 
        layer_wg_etch
    ):
    """
    Add a post-array grating and a ring resonator.
    
    Args:
        cell (gdspy.cell): A gdspy cell in which the grating will be added.
        ring_radius (float): Radius of the ring nearby the grating in um, measured at center of the waveguide.
        ring_width (float): Width of the ring nearby the grating in um.
        etch_width (float): Width of the etching regions in each side of the ring in um.
        post_period (float): Angular period of posts that form the grating.
        post_num (int): Number of posts that form the grating.
        post_gap (float): Separation distance between the ring and the post array in um.
        post_radius (float): Radius of the posts in um.
        post_angle_offset (float): Angular offset of the post array in radian.
        ring_center (tuple): Position of the ring in um.
        tolerance (float): Approximate curvature resolution of the ring and post.
        layer_wg (int): Layer index of the grating and the ring.
        layer_wg_etch (int): Layer index of the etch regions around the grating and the ring.

    Returns:
        gdspy.cell: A gdspy cell with the grating in ring.
    """
    # Define ring resonator
    ring = gdspy.Round(
        center=ring_center, 
        radius=ring_radius+ring_width/2, 
        inner_radius=ring_radius-ring_width/2, 
        tolerance=tolerance,
        layer=layer_wg
    )
    ring_etch = gdspy.Round(
        center=ring_center, 
        radius=ring_radius+ring_width/2+etch_width, 
        inner_radius=ring_radius-ring_width/2-etch_width, 
        tolerance=tolerance,
        layer=layer_wg_etch
    )
    cell.add(ring)
    cell.add(ring_etch)

    # Define post-array grating
    post_array_radius = ring_radius + ring_width/2 + post_gap + post_radius # Radius of the array of posts
    for idx_post in range(post_num):
        theta = post_angle_offset + post_period * idx_post # Azimuth angle of the post
        post_center = (ring_center[0] + post_array_radius * np.cos(theta), 
            ring_center[1] + post_array_radius * np.sin(theta))
        post = gdspy.Round(center=post_center, radius=post_radius, tolerance=tolerance, max_points=0, layer=layer_wg)
        cell.add(post)
    return cell

def spiral_waveguide_dissipation(
        cell,
        spiral_begin_point,
        min_bend_radius,
        adiabaticity,
        number_of_points,
        wg_width,
        min_wg_width,
        etch_width,
        orientation,
        layer_wg,
        layer_wg_etch, 
        layer_electrode,
        mirror_line=None
    ):
    """
    Add an Archimedean-spiral-shaped waveguide to dissipate light.

    Args:
        cell (gdspy.cell): A gdspy cell in which the spiral waveguide will be added.
        spiral_begin_point (tuple): Begin point of the spiral waveguide in um.
        min_bend_radius (float): Minimum curvature radius of the spiral waveguide in um.
        adiabaticity (float): This parameter determines the length of the spiral.
        number_of_points (int): Number of points used to approximate a waveguide bend and a spiral.
        wg_width (float): Width of the spiral waveguide in um, achieved at the beginning of the spiral.
        min_wg_width (float): Minimum width of the spiral waveguide in um, achieved at the terminal of the spiral.
        etch_width (float): Width of the exposure regions in both sides of the spiral waveguide in um.
        orientation (float): Orientation angle of the spiral waveguide at its beginning in radian.
        layer_wg (int): Layer index of the spiral waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the spiral waveguide.
        layer_electrode (int): Layer index of the heater.
        mirror_line (str): 'y' for mirror the spiral waveguide over y-axis, default to `None`.

    Returns:
        gdspy.cell: A gdspy cell with the spiral waveguide.
    """
    # Add an Archimedean-spiral-shaped waveguide
    spiral_angular_range = 5 / 2 * np.pi
    spiral_curve = geometry.euler_spiral(min_bend_radius, spiral_angular_range, number_of_points)
    spiral_curve[:, 0] = spiral_curve[:, 0] + spiral_begin_point[0]
    spiral_curve[:, 1] = spiral_curve[:, 1] + spiral_begin_point[1]
    spiral_curve = geometry.rotate_array(spiral_curve, spiral_begin_point, orientation)
    spiral_curve = geometry.scale_array(spiral_curve, spiral_begin_point, adiabaticity)
    spiral_wg = gdspy.PolyPath(
        spiral_curve, 
        width=np.linspace(wg_width, min_wg_width, len(spiral_curve)),
        number_of_paths=len(spiral_curve),
        layer=layer_wg, 
        ends='flush', 
    )
    spiral_etch = gdspy.PolyPath(
        spiral_curve, 
        wg_width+etch_width*2,
        number_of_paths=1,
        layer=layer_wg_etch, 
        ends='flush', 
    )
    if mirror_line == 'y':
        cell.add(spiral_wg.mirror((spiral_begin_point[0], 0), (spiral_begin_point[0], 1)))
        cell.add(spiral_etch.mirror((spiral_begin_point[0], 0), (spiral_begin_point[0], 1)))
    else:
        cell.add(spiral_wg)
        cell.add(spiral_etch)

    # Add heater metal on the spiral waveguide to absorb light
    spiral_center = np.mean(spiral_curve[len(spiral_curve)//2:, :], axis=0)
    circular_metal = gdspy.Round(
        center=spiral_center, 
        radius=min_bend_radius*adiabaticity*2.5, 
        layer=layer_electrode
    )
    circular_etch = gdspy.Round(
        center=spiral_center, 
        radius=min_bend_radius*adiabaticity*2.5+etch_width, 
        layer=layer_wg_etch
    )
    if mirror_line == 'y':
        cell.add(circular_metal.mirror((spiral_begin_point[0], 0), (spiral_begin_point[0], 1)))
        cell.add(circular_etch.mirror((spiral_begin_point[0], 0), (spiral_begin_point[0], 1)))
    else:
        cell.add(circular_metal)
        cell.add(circular_etch)

    return cell

def spiral_waveguide(
        cell, 
        spiral_begin_point, 
        input_offset_y, 
        output_offset_y, 
        straight_length, 
        min_bend_radius, 
        max_bend_radius, 
        num_circle, 
        number_of_points, 
        wg_width, 
        etch_width, 
        layer_wg, 
        layer_wg_etch
    ):
    """
    Add a spiral waveguide with euler bends.

    Args:
        cell (gdspy.cell): A gdspy cell in which the spiral waveguide will be added.
        spiral_begin_point (tuple): Begin point of the spiral waveguide in um.
        input_offset_y (float): Offset of the input section along y-axis in um.
        output_offset_y (float): Offset of the output section along y-axis in um.
        straight_length (float): Length of the straight section in um.
        min_bend_radius (float): Minimum curvature radius of the spiral waveguide in um.
        max_bend_radius (float): Maximum curvature radius of the spiral waveguide in um.
        num_circle (int): Number of circles in the spiral waveguide.
        number_of_points (int): Number of points used to approximate a waveguide bend.
        wg_width (float): Width of the spiral waveguide in um.
        etch_width (float): Width of the exposure regions in both sides of the spiral waveguide in um.
        layer_wg (int): Layer index of the spiral waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the spiral waveguide.
    
    Returns:
        gdspy.cell: A gdspy cell with the spiral waveguide.
        tuple: A tuple of the end point of the spiral waveguide.
        float: The length of the spiral waveguide.
    """
    # Define curvature radii of euler bends in the spiral waveguide
    bend_radii = np.linspace(min_bend_radius, max_bend_radius, num=(num_circle+1))

    # Define the input section of the spiral waveguide
    input_bend = geometry.euler_curve(bend_radii[-1], np.pi/2, number_of_points)
    input_bend[:, 1] = -input_bend[:, 1] # Mirror the input bend over the x-axis
    input_bend[:, 0] = input_bend[:, 0] + spiral_begin_point[0] # Translate the input bend to the begin point
    input_bend[:, 1] = input_bend[:, 1] + spiral_begin_point[1] # Translate the input bend to the begin point
    input_section_end = np.array([input_bend[-1, 0], input_bend[-1, 1]-input_offset_y]).reshape(1, 2)
    spiral_curve = np.concatenate((input_bend, input_section_end), axis=0)

    # Define a spiral curve that goes from outer side to inner side, namely outer spiral curve
    for idx_bend in range(num_circle):
        if idx_bend % 2 == 0:
            # Add a straight section
            straight_section_end = np.array([spiral_curve[-1, 0], spiral_curve[-1, 1]-straight_length])
            spiral_curve = np.concatenate((spiral_curve, straight_section_end.reshape(1, 2)), axis=0)
            # Add a bend section
            spiral_bend = geometry.euler_curve(bend_radii[-1-idx_bend], np.pi, number_of_points)
            # Inverse the bend over the origin point and mirror the bend over line y = x
            spiral_bend = np.stack([-spiral_bend[:, 1], -spiral_bend[:, 0]], axis=1)
            spiral_bend[:, 0] = spiral_bend[:, 0] + spiral_curve[-1, 0]
            spiral_bend[:, 1] = spiral_bend[:, 1] + spiral_curve[-1, 1]
            spiral_curve = np.concatenate((spiral_curve, spiral_bend[1:, :]), axis=0)
        elif idx_bend % 2 == 1:
            # Add a straight section
            straight_section_end = np.array([spiral_curve[-1, 0], spiral_curve[-1, 1]+straight_length])
            spiral_curve = np.concatenate((spiral_curve, straight_section_end.reshape(1, 2)), axis=0)
            # Add a bend section
            spiral_bend = geometry.euler_curve(bend_radii[-1-idx_bend], np.pi, number_of_points)
            spiral_bend = np.stack([spiral_bend[:, 1], spiral_bend[:, 0]], axis=1) # Mirror the bend over line y = x
            spiral_bend[:, 0] = spiral_bend[:, 0] + spiral_curve[-1, 0]
            spiral_bend[:, 1] = spiral_bend[:, 1] + spiral_curve[-1, 1]
            spiral_curve = np.concatenate((spiral_curve, spiral_bend[1:, :]), axis=0)
        else:
            raise ValueError('`idx_bend` must be an integer.')
    outer_spiral_end = tuple(spiral_curve[-1, :])

    # Define a spiral curve that goes from inner side to outer side, namely inner spiral curve
    spiral_gap = np.pi*(bend_radii[-1]-bend_radii[-2])/2*0.9 # Approximate gap between the inner and outer spiral curves
    spiral_curve_inner = np.array([input_section_end[0, 0]-spiral_gap, input_section_end[0, 1]]).reshape(1, 2)
    for idx_bend in range(num_circle-1):
        if idx_bend % 2 == 0:
            # Add a straight section
            straight_section_end = np.array([spiral_curve_inner[-1, 0], spiral_curve_inner[-1, 1]-straight_length])
            spiral_curve_inner = np.concatenate((spiral_curve_inner, straight_section_end.reshape(1, 2)), axis=0)
            # Add a bend section
            spiral_bend = geometry.euler_curve(bend_radii[-2-idx_bend], np.pi, number_of_points)
            # Inverse the bend over the origin point and mirror the bend over line y = x
            spiral_bend = np.stack([-spiral_bend[:, 1], -spiral_bend[:, 0]], axis=1)
            spiral_bend[:, 0] = spiral_bend[:, 0] + spiral_curve_inner[-1, 0]
            spiral_bend[:, 1] = spiral_bend[:, 1] + spiral_curve_inner[-1, 1]
            spiral_curve_inner = np.concatenate((spiral_curve_inner, spiral_bend[1:, :]), axis=0)
        elif idx_bend % 2 == 1:
            # Add a straight section
            straight_section_end = np.array([spiral_curve_inner[-1, 0], spiral_curve_inner[-1, 1]+straight_length])
            spiral_curve_inner = np.concatenate((spiral_curve_inner, straight_section_end.reshape(1, 2)), axis=0)
            # Add a bend section
            spiral_bend = geometry.euler_curve(bend_radii[-2-idx_bend], np.pi, number_of_points)
            spiral_bend = np.stack([spiral_bend[:, 1], spiral_bend[:, 0]], axis=1) # Mirror the bend over line y = -x
            spiral_bend[:, 0] = spiral_bend[:, 0] + spiral_curve_inner[-1, 0]
            spiral_bend[:, 1] = spiral_bend[:, 1] + spiral_curve_inner[-1, 1]
            spiral_curve_inner = np.concatenate((spiral_curve_inner, spiral_bend[1:, :]), axis=0)
        else:
            raise ValueError('`idx_bend` must be an integer.')
    inner_spiral_end = tuple(spiral_curve_inner[-1, :])

    # Define the S-shaped euler spiral curve at the center
    center_spiral = geometry.euler_curve_S_connection(outer_spiral_end, inner_spiral_end, np.pi/3, 'y', number_of_points)

    # Define the output section of the spiral waveguide
    spiral_bend_start = spiral_curve_inner[0, :].reshape(1, 2)
    spiral_bend = geometry.euler_curve(bend_radii[-1], np.pi, number_of_points)
    spiral_bend = np.stack([-spiral_bend[:, 1], spiral_bend[:, 0]], axis=1) # Mirror the bend over line y = -x
    spiral_bend[:, 0] = spiral_bend[:, 0] + spiral_bend_start[0, 0]
    spiral_bend[:, 1] = spiral_bend[:, 1] + spiral_bend_start[0, 1]
    spiral_curve_inner = np.concatenate((np.flip(spiral_curve_inner, axis=0), spiral_bend), axis=0)
    output_section_start = np.array([spiral_bend[-1, 0], spiral_bend[-1, 1]-straight_length-output_offset_y]).reshape(1, 2)
    output_bend = geometry.euler_curve(bend_radii[-1], np.pi/2, number_of_points)
    output_bend = np.stack([output_bend[:, 1], -output_bend[:, 0]], axis=1) # Mirror the bend over line y = x and flip over x-axis
    output_bend[:, 0] = output_bend[:, 0] + output_section_start[0, 0]
    output_bend[:, 1] = output_bend[:, 1] + output_section_start[0, 1]
    spiral_curve_inner = np.concatenate((spiral_curve_inner, output_bend), axis=0)

    # Merge the inner and outer spiral curves
    spiral_curve = np.concatenate((spiral_curve, center_spiral[1:-1, :], spiral_curve_inner), axis=0)
    spiral_end_point = tuple(spiral_curve[-1, :])

    # Calculate the total length of the spiral curves
    spiral_length = geometry.curve_length(spiral_curve)

    # Add spiral waveguide to cell
    spiral_wg = gdspy.FlexPath(spiral_curve, wg_width, layer=layer_wg)
    spiral_wg_etch = gdspy.FlexPath(spiral_curve, wg_width+2*etch_width, layer=layer_wg_etch)
    cell.add(spiral_wg)
    cell.add(spiral_wg_etch)

    return cell, spiral_end_point, spiral_length

def electrode(cell, electrode_length, electrode_width, electrode_gap, orientation, center, layer_electrode):
    """
    Add two electrodes around a waveguide.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode will be added.
        electrode_length (float): Length of the electrode in um.
        electrode_width (float): Width of the electrode in um.
        electrode_gap (float): Separation distance between the electrode and the waveguide in um.
        orientation (float): Orientation angle of the electrode in radian.
        center (tuple): Position of the electrode in um.
        layer_electrode (int): Layer index of the electrode.
    
    Returns:
        gdspy.cell: A gdspy cell with the electrode.
    """
    # Add the first electrode
    electrode_center_1 = (center[0], center[1] + electrode_gap + electrode_width/2)
    electrode_1 = gdspy.Rectangle(
        (-electrode_length/2+electrode_center_1[0], -electrode_width/2+electrode_center_1[1]),
        (electrode_length/2+electrode_center_1[0], electrode_width/2+electrode_center_1[1]),
        layer=layer_electrode
    )
    cell.add(electrode_1.rotate(orientation, center))

    # Add the second electrode
    electrode_center_2 = (center[0], center[1] - electrode_gap - electrode_width/2)
    electrode_2 = gdspy.Rectangle(
        (-electrode_length/2+electrode_center_2[0], -electrode_width/2+electrode_center_2[1]),
        (electrode_length/2+electrode_center_2[0], electrode_width/2+electrode_center_2[1]),
        layer=layer_electrode
    )
    cell.add(electrode_2.rotate(orientation, center))

    return cell

def electrode_ring(cell, ring_radius, ring_width, center, gap, gap_pos, layer_electrode, layer_wg_etch):
    """
    Add a ring-shaped electrode.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode will be added.
        ring_radius (float): Radius of the ring in um.
        ring_width (float): Width of the ring in um.
        center (tuple): Position of the ring in um.
        gap (float): Gap in the ring in um.
        gap_pos (float): Angular position of the gap in radian.
        layer_electrode (int): Layer index of the electrode.
        layer_wg_etch (int): Layer index of the etch regions around the ring.

    Returns:
        gdspy.cell: A gdspy cell with the electrode.
    """
    # Add the ring-shaped electrode
    gap_angular_range = gap / ring_radius
    electrode = gdspy.Round(
        center=center, 
        radius=ring_radius+ring_width/2, 
        inner_radius=ring_radius-ring_width/2, 
        initial_angle = gap_pos+gap_angular_range/2, 
        final_angle = gap_pos-gap_angular_range/2+2*np.pi, 
        layer=layer_electrode
    )
    cell.add(electrode)
    electrode = gdspy.Round(
        center=center, 
        radius=ring_radius+ring_width/2, 
        inner_radius=ring_radius-ring_width/2, 
        initial_angle = gap_pos+gap_angular_range/2, 
        final_angle = gap_pos-gap_angular_range/2+2*np.pi, 
        layer=layer_wg_etch
    )
    cell.add(electrode)
    return cell

def electrode_ring_probe(
        cell, 
        bridge_height, 
        bridge_width, 
        bridge_gap, 
        ring_radius, 
        ring_width, 
        ring_gap, 
        ring_center, 
        pad_height, 
        pad_heater_gap, 
        gap_pos, 
        number_of_points, 
        layer_heater, 
        layer_probe, 
        layer_wg_etch
    ):
    """
    Add an electrode connecting the ring-shaped heater and the square-shaped contact pads.
    
    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode will be added.
        bridge_height (float): Height of the bridge in um.
        bridge_width (float): Width of the bridge in um.
        bridge_gap (float): Gap between the bridge and the ring in um.
        ring_radius (float): Radius of the ring-shaped heater in um.
        ring_width (float): Width of the ring-shaped heater in um.
        ring_gap (float): Gap between the ring-shaped heater and the bridge in um.
        ring_center (tuple): Position of the ring-shaped heater in um.
        pad_height (float): Height of the contact pads in um.
        pad_heater_gap (float): Gap between the probe contact metal and the heater metal in um.
        gap_pos (float): Angular position of the gap in radian.
        number_of_points (int): Number of points used to approximate the arc-shaped edge.
        layer_heater (int): Layer index of the heater metal.
        layer_probe (int): Layer index of the contact pads.
        layer_wg_etch (int): Layer index of the etch regions around the bridge.
    """
    bridge_gap_angular_range_min = ring_gap / ring_radius
    bridge_angular_range_min = (ring_gap + ring_width * 2) / ring_radius

    """
    Warning: The boolean operation of gdspy may not work properly when the two polygons are too close to each other.
    For example, when using the `note` operation, even if two objects are identical, the result may not be empty.
    """
    boolean_operation_correction = 0.001

    # Define the bridge gap edge, heater metal in the enclosed region will be lifted off 
    edge_param = np.linspace(-0.5, 0.5, number_of_points)
    bridge_gap_edge_outer = np.stack(
        [np.cos(gap_pos + edge_param * bridge_gap_angular_range_min) * (ring_radius - ring_width / 2), 
        np.sin(gap_pos + edge_param * bridge_gap_angular_range_min) * (ring_radius - ring_width / 2)], 
        axis=-1
    ) + boolean_operation_correction * np.array([np.cos(gap_pos), np.sin(gap_pos)])
    bridge_gap_edge_inner = np.array([
        ring_radius - bridge_height, bridge_gap / 2, 
        ring_radius - bridge_height, -bridge_gap / 2
    ]).reshape(-1, 2) - boolean_operation_correction
    bridge_gap_edge_inner = geometry.rotate_array(bridge_gap_edge_inner, [0, 0], gap_pos)
    bridge_gap_edge = np.concatenate((bridge_gap_edge_outer, bridge_gap_edge_inner), axis=0)
    bridge_gap_edge = bridge_gap_edge + np.array(ring_center)
    _bridge_gap = gdspy.Polygon(bridge_gap_edge, layer=layer_heater)
    _bridge_gap_etch = gdspy.Polygon(bridge_gap_edge, layer=layer_wg_etch)

    # Define the bridge, heater metal in the enclosed region will be kept
    bridge_edge_outer = np.stack(
        [np.cos(gap_pos + edge_param * bridge_angular_range_min) * (ring_radius - ring_width / 2), 
        np.sin(gap_pos + edge_param * bridge_angular_range_min) * (ring_radius - ring_width / 2)], 
        axis=-1
    )
    bridge_edge_inner = np.array([
        ring_radius - bridge_height, bridge_width + bridge_gap / 2, 
        ring_radius - bridge_height, -bridge_width - bridge_gap / 2
    ]).reshape(-1, 2)
    bridge_edge_inner = geometry.rotate_array(bridge_edge_inner, [0, 0], gap_pos)
    bridge_edge = np.concatenate((bridge_edge_outer, bridge_edge_inner), axis=0)
    bridge_edge = bridge_edge + np.array(ring_center)
    _bridge = gdspy.Polygon(bridge_edge, layer=layer_heater)
    _bridge_etch = gdspy.Polygon(bridge_edge, layer=layer_wg_etch)

    # Boolean operation to remove bridge gap from bridge
    bridge = gdspy.boolean(_bridge, _bridge_gap, 'not', layer=layer_heater)
    bridge_etch = gdspy.boolean(_bridge_etch, _bridge_gap_etch, 'not', layer=layer_wg_etch)

    cell.add(bridge)
    cell.add(bridge_etch)

    # Define the probe metal overlap with the heater metal
    circular_mask = gdspy.Round(ring_center, ring_radius-ring_width/2-pad_heater_gap, layer=layer_probe)
    heater_probe_metal_overlap = gdspy.boolean(circular_mask, bridge, 'and', layer=layer_probe)
    cell.add(heater_probe_metal_overlap)

    # Define the contact pads
    contact_pad_pts = [
        bridge_edge_inner[0], 
        bridge_edge_inner[1], 
        bridge_edge_inner[1]+geometry.rotate([0, 0], [-pad_height, 0], gap_pos), 
        bridge_edge_inner[0]+geometry.rotate([0, 0], [-pad_height, 0], gap_pos)
    ]
    contact_pad_pts = contact_pad_pts + np.array(ring_center)
    _contact_pad = gdspy.Polygon(contact_pad_pts, layer=layer_probe)
    bridge_gap_edge_inner = np.array([
        ring_radius - bridge_height, bridge_gap / 2, 
        ring_radius - bridge_height, -bridge_gap / 2
    ]).reshape(-1, 2) # Redefine the bridge gap inner edge without the correction for boolean operation
    bridge_gap_edge_inner = geometry.rotate_array(bridge_gap_edge_inner, [0, 0], gap_pos)
    contact_pad_gap_pts = [
        bridge_gap_edge_inner[0]+geometry.rotate([0, 0], [boolean_operation_correction, 0], gap_pos), 
        bridge_gap_edge_inner[1]+geometry.rotate([0, 0], [boolean_operation_correction, 0], gap_pos), 
        bridge_gap_edge_inner[1]+geometry.rotate([0, 0], [-pad_height-boolean_operation_correction, 0], gap_pos), 
        bridge_gap_edge_inner[0]+geometry.rotate([0, 0], [-pad_height-boolean_operation_correction, 0], gap_pos)
    ]
    contact_pad_gap_pts = contact_pad_gap_pts + np.array(ring_center)
    contact_pad_gap = gdspy.Polygon(contact_pad_gap_pts, layer=layer_probe)
    contact_pad = gdspy.boolean(_contact_pad, contact_pad_gap, 'not', layer=layer_probe)
    contact_pad_etch = gdspy.boolean(_contact_pad, contact_pad_gap, 'not', layer=layer_wg_etch)

    cell.add(contact_pad)
    cell.add(contact_pad_etch)

    return cell

def electrode_probe(
        cell, 
        probe_height, 
        probe_width, 
        probe_gap, 
        bridge_height, 
        bridge_width, 
        bridge_gap, 
        probe_heater_gap, 
        orientation, 
        center, 
        layer_heater, 
        layer_probe, 
        layer_wg_etch
    ):
    """
    Add a probe contact pad.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode will be added.
        probe_height (float): Length of the probe electrode in um.
        probe_width (float): Width of the probe electrode in um.
        probe_gap (float): Separation distance between the probe electrode and the waveguide in um.
        bridge_height (float): Height of the bridge in um.
        bridge_width (float): Width of the bridge in um.
        bridge_gap (float): Gap between the bridges in each side in um.
        probe_heater_gap (float): Gap between the probe contact metal and the heater in um.
        orientation (float): Orientation angle of the probe electrode in radian.
        center (tuple): Position of the probe electrode in um.
        layer_heater (int): Layer index of the heater metal.
        layer_probe (int): Layer index of the probe electrode.
        layer_wg_etch (int): Layer index of the etch regions around the electrode.
    
    Returns:
        gdspy.cell: A gdspy cell with the electrode.
    """
    # Add a bridge electrode connecting the probe pad and the straight heater
    bridge_edge = np.array([
        -bridge_gap/2-bridge_width, 0, bridge_gap/2+bridge_width, 0, 
        probe_gap/2+probe_width, bridge_height, -probe_gap/2-probe_width, bridge_height
    ]).reshape(-1, 2)
    bridge_edge = bridge_edge + np.array(center)
    _bridge = gdspy.Polygon(bridge_edge, layer=layer_heater)
    _bridge_etch = gdspy.Polygon(bridge_edge, layer=layer_wg_etch)

    # Define a region between the electrode bridges, where metal will be lifted off
    bridge_gap_edge = np.array([
        -bridge_gap/2, 0, bridge_gap/2, 0, 
        probe_gap/2, bridge_height, -probe_gap/2, bridge_height
    ]).reshape(-1, 2)
    bridge_gap_edge = bridge_gap_edge + np.array(center)
    _bridge_gap = gdspy.Polygon(bridge_gap_edge, layer=layer_heater)
    _bridge_gap_etch = gdspy.Polygon(bridge_gap_edge, layer=layer_wg_etch)

    bridge = gdspy.boolean(_bridge, _bridge_gap, 'not', layer=layer_heater).rotate(orientation, center)
    bridge_etch = gdspy.boolean(_bridge_etch, _bridge_gap_etch, 'not', layer=layer_wg_etch).rotate(orientation, center)

    cell.add(bridge)
    cell.add(bridge_etch)

    # Define probe contact metals overlap with the heater metal
    mask_probe = gdspy.Rectangle(
        [-bridge_gap/2-bridge_width, 0], [bridge_gap/2+bridge_width, probe_heater_gap], layer=layer_probe
    ).translate(center[0], center[1]).rotate(orientation, center)
    probe_bridge_overlap = gdspy.boolean(bridge, mask_probe, 'not', layer=layer_probe)
    cell.add(probe_bridge_overlap)

    # Add the probe contact pads
    probe_center = np.array([center[0], center[1] + probe_height / 2 + bridge_height])
    probe_edge = np.array([
        -probe_width-probe_gap/2, -probe_height/2, probe_width+probe_gap/2, -probe_height/2, 
        probe_width+probe_gap/2, probe_height/2, -probe_width-probe_gap/2, probe_height/2
    ]).reshape(-1, 2)
    probe_edge = probe_edge + probe_center
    _probe = gdspy.Polygon(probe_edge, layer=layer_probe)
    _probe_etch = gdspy.Polygon(probe_edge, layer=layer_wg_etch)

    # Define the region between the probe pads, where metal will be lifted off
    probe_gap_edge = np.array([
        -probe_gap/2, -probe_height/2, probe_gap/2, -probe_height/2, 
        probe_gap/2, probe_height/2, -probe_gap/2, probe_height/2
    ]).reshape(-1, 2)
    probe_gap_edge = probe_gap_edge + probe_center
    _probe_gap = gdspy.Polygon(probe_gap_edge, layer=layer_probe)
    _probe_gap_etch = gdspy.Polygon(probe_gap_edge, layer=layer_wg_etch)

    probe = gdspy.boolean(_probe, _probe_gap, 'not', layer=layer_probe).rotate(orientation, center)
    probe_etch = gdspy.boolean(_probe_etch, _probe_gap_etch, 'not', layer=layer_wg_etch).rotate(orientation, center)

    cell.add(probe)
    cell.add(probe_etch)

    return cell

def electrode_by_racetrack(
        cell, 
        electrode_length, 
        electrode_width,
        electrode_gap, 
        coupling_length, 
        coupler_bend_radius, 
        coupler_bend_orientation, 
        racetrack_radius, 
        wg_width, 
        gap, 
        center, 
        layer_electrode, 
        bend_type='euler'
    ):
    """
    Add two electrodes on an arm of a loaded racetrack resonator.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode will be added.
        electrode_length (float): Length of the electrode in um.
        electrode_width (float): Width of the electrode in um.
        electrode_gap (float): Separation distance between the electrode and the racetrack in um.
        coupling_length (float): Length of the coupling region in um.
        coupler_bend_radius (float): Minimum curvature radius of the bend section of the coupler in um.
        coupler_bend_orientation (float): Change in the orientation angle across the bend section of the coupler in radian.
        racetrack_radius (float): Minimum curvature radius of the racetrack in um.
        number_of_points (int): Number of points used to approximate the racetrack bend.
        wg_width (float): Width of the waveguide and racetrack in um.
        gap (float): Separation distance between the racetrack and the straight waveguide in um.
        center (tuple): Position of the loaded racetrack resonator in um.
        layer_electrode (int): Layer index of the electrode.
        bend_type (str): 'euler'/'circular' for euler-spiral/circular bend section, default to `'euler'`.

    Returns:
        gdspy.cell: A gdspy cell with the electrode.
    """
    # Determine the position for adding the electrodes
    number_of_points = 64 # Number of points used to approximate the euler bends
    coupler_curve = geometry.euler_curve_protrusion(coupling_length, coupler_bend_radius, coupler_bend_orientation, number_of_points)
    coupler_size = [max(np.abs(coupler_curve[:, 0])), max(np.abs(coupler_curve[:, 1]))] # Size of the coupler
    if bend_type == 'euler':
        euler_bend = geometry.euler_curve(racetrack_radius, np.pi, number_of_points) # Euler bend in the racetrack
        racetrack_height = euler_bend[-1][1] # Height of the racetrack
    elif bend_type == 'circular':
        semicircular_bend = geometry.circular_arc(racetrack_radius, np.pi, number_of_points) # Semicircular bend in the racetrack
        racetrack_height = semicircular_bend[-1][1] # Height of the racetrack
    electrode_x = center[0]
    electrode_y = center[1] + coupler_size[1] + wg_width + gap + racetrack_height
    electrode_center = (electrode_x, electrode_y)

    # Add the electrodes
    cell = electrode(cell, electrode_length, electrode_width, electrode_gap, 0, electrode_center, layer_electrode)
    return cell

def electrode_pad(cell, pad_width, pad_height, center, layer_electrode, layer_wg_etch):
    """
    Add a rectangular electrode pad for electrical contact.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode pad will be added.
        pad_width (float): Width of the electrode pad in um.
        pad_height (float): Height of the electrode pad in um.
        center (tuple): Position of the electrode pad in um.
        layer_electrode (int): Layer index of the electrode pad.
        layer_wg_etch (int): Layer index of the etch regions around the electrode pad.

    Returns:
        gdspy.cell: A gdspy cell with the electrode pad.
    """
    # Add the electrode pad
    pad = gdspy.Rectangle(
        (-pad_width/2+center[0], -pad_height/2+center[1]),
        (pad_width/2+center[0], pad_height/2+center[1]),
        layer=layer_electrode
    )
    cell.add(pad)
    pad_etch = gdspy.Rectangle(
        (-pad_width/2+center[0], -pad_height/2+center[1]),
        (pad_width/2+center[0], pad_height/2+center[1]),
        layer=layer_wg_etch
    )
    cell.add(pad_etch)
    return cell

def electrode_taper(cell, taper_length, taper_width1, taper_width2, center, angle, layer_electrode):
    """
    Add a tapered electrode for electrical contact.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode will be added.
        taper_length (float): Length of the electrode in um.
        taper_width1 (float): Width of the electrode at the beginning in um.
        taper_width2 (float): Width of the electrode at the end in um.
        center (tuple): Center of the beginning line of the taper.
        angle (float): Orientation angle of the electrode in radian.
        layer_electrode (int): Layer index of the electrode.

    Returns:
        gdspy.cell: A gdspy cell with the electrode.
    """
    # Add the tapered electrode
    electrode = gdspy.Path(taper_width1, center)
    electrode.segment(length=taper_length, direction=angle, final_width=taper_width2, layer=layer_electrode)
    cell.add(electrode)
    return cell

def metal_wire(cell, wire_width, wire_height, center, layer_electrode):
    """
    Add a rectangular metal wire for electrical connection.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode pad will be added.
        wire_width (float): Width of the electrode pad in um.
        wire_height (float): Height of the electrode pad in um.
        center (tuple): Position of the electrode pad in um.
        layer_electrode (int): Layer index of the electrode pad.s

    Returns:
        gdspy.cell: A gdspy cell with the electrode pad.
    """
    # Add the electrode pad
    cell = electrode_pad(cell, wire_width, wire_height, center, layer_electrode)
    return cell

def electrode_on_racetrack(
        cell, 
        wg_width, 
        racetrack_length, 
        racetrack_radius, 
        racetrack_width, 
        gap, 
        coupling_length, 
        bend_orientation, 
        bend_radius, 
        etch_width, 
        center, 
        number_of_points, 
        electrode_heater_width, 
        electrode_gap, 
        electrode_connect_width, 
        electrode_pad_width, 
        electrode_pad_height, 
        electrode_pad_center, 
        layer_heater, 
        layer_probe, 
        layer_probe_auxiliary, 
        bend_type='euler', 
        flip=False, 
        bypass_coupler=False, 
        bypass_electrode_size=None
    ):
    """
    Add a racetrack-shaped electrode on a loaded racetrack resonator.

    Args:
        cell (gdspy.cell): A gdspy cell in which the electrode will be added.
        wg_width (float): Width of the racetrack resonator in um.
        racetrack_length (float): Length of the racetrack in um.
        racetrack_radius (float): Minimum curvature radius of the racetrack in um.
        racetrack_width (float): Width of the racetrack in um.
        gap (float): Separation distance between the racetrack and the straight waveguide in um.
        coupling_length (float): Length of the coupling region in um.
        bend_orientation (float): Change in the orientation angle across the euler-spiral bend section in radian.
        bend_radius (float): Minimum curvature radius of the euler waveguide bend coupler in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide/racetrack in um.
        center (tuple): Position of the loaded racetrack resonator in um.
        number_of_points (int): Number of points used to approximate the racetrack bend.
        electrode_heater_width (float): Width of the metallic microheater in um.
        electrode_gap (float): Separation distance between the two ends of the electrode in um.
        electrode_connect_width (float): Width of the electrode that connects the electrode pad and heater in um.
        electrode_pad_width (float): Width of the electrode pad in um.
        electrode_pad_height (float): Height of the electrode pad in um.
        electrode_pad_center (tuple): Relative position of the electrode pad to right end point of top straight section of racetrack in um.
        layer_heater (int): Layer index of the heater.
        layer_probe (int): Layer index of the probe for electrical contact.
        layer_probe_auxiliary (int): Layer index of the auxiliary electrode, to be removed through bool operation.
        bend_type (str): 'euler'/'circular' for euler-spiral/circular bend section, default to `'euler'`.
        flip (bool): Whether to flip the electrode to the left side of the racetrack, default to `False`.
        bypass_coupler (bool): Whether to bypass the coupler, activated only for euler bend type, default to `False`.
        bypass_electrode_size (tuple): Size of the rectangular electrode that bypasses the coupler, default to `None`.

    Returns:
        gdspy.cell: A gdspy cell with the electrode.
    """
    # Determine the size of the waveguide coupler
    coupler_size = pulley_coupler(
        cell, 
        wg_width, 
        etch_width, 
        coupling_length, 
        bend_radius, 
        bend_orientation, 
        0, 
        center, 
        number_of_points, 
        0, 
        0, 
        return_size_only=True
    )

    # Add the electrode
    if bend_type == 'euler':
        euler_bend = geometry.euler_curve(racetrack_radius, np.pi, number_of_points) # Euler bend in the racetrack
        racetrack_height = euler_bend[-1][1] # Height of the racetrack
        racetrack_center = (center[0], center[1] + wg_width/2 + gap + racetrack_width/2 + racetrack_height/2 + coupler_size[1])

        # Add racetrack electrode
        if not bypass_coupler:
            protrusion=(0, 0)
        elif bypass_coupler and (bypass_electrode_size is None):
            raise ValueError('`bypass_electrode_size` must be specified when `bypass_coupler` is True.')
        elif bypass_coupler and (bypass_electrode_size is not None):
            protrusion=bypass_electrode_size # Size of rectangular electrode that bypasses the coupler 
        cell = racetrack(
            cell, 
            racetrack_length, 
            racetrack_radius, 
            electrode_heater_width, 
            0, # No clad for electrode, set clad width to zero
            racetrack_center, 
            number_of_points, 
            layer_heater, 
            layer_heater, 
            protrusion=protrusion
        )

        # Calculate the length of the straight waveguide in racetrack
        euler_bend_length = 2 * np.pi * racetrack_radius # Length of the euler bend
        straight_length = racetrack_length / 2 - euler_bend_length # Length of the straight waveguide
    elif bend_type == 'circular':
        semicircular_bend = geometry.circular_arc(racetrack_radius, np.pi, number_of_points) # Semicircular bend in the racetrack
        racetrack_height = semicircular_bend[-1][1] # Height of the racetrack
        racetrack_center = (center[0], center[1] + wg_width/2 + gap + racetrack_width/2 + racetrack_height/2 + coupler_size[1])

        # Add racetrack electrode
        cell = racetrack_circular(
            cell, 
            racetrack_length, 
            racetrack_radius, 
            electrode_heater_width, 
            0, # No clad for electrode, set clad width to zero
            racetrack_center, 
            number_of_points, 
            layer_heater, 
            layer_heater
        )

        # Calculate the length of the straight waveguide in racetrack
        euler_bend_length = np.pi * racetrack_radius # Length of the semicircular bend
        straight_length = racetrack_length / 2 - euler_bend_length # Length of the straight waveguide
    else:
        raise ValueError('`bend_type` must be `euler` or `circular`.')
    
    # Add electrode pads
    if flip:
        # Define position of the electrode pad on the left
        pad_reference = (racetrack_center[0]-straight_length/2, racetrack_center[1]+racetrack_height/2)
        # Center of the electrode pad on the right
        pad_center1 = (pad_reference[0]-electrode_pad_center[0], pad_reference[1]+electrode_pad_center[1])
        # Center of the electrode pad on the left
        pad_center2 = (pad_center1[0]+electrode_pad_width+electrode_gap+2*electrode_connect_width, pad_center1[1])
    elif not flip:
        # Define position of the electrode pad on the right
        pad_reference = (racetrack_center[0]+straight_length/2, racetrack_center[1]+racetrack_height/2)
        # Center of the electrode pad on the right
        pad_center1 = (pad_reference[0]+electrode_pad_center[0], pad_reference[1]+electrode_pad_center[1])
        # Center of the electrode pad on the left
        pad_center2 = (pad_center1[0]-electrode_pad_width-electrode_gap-2*electrode_connect_width, pad_center1[1])
    else:
        raise ValueError('`flip` must be `True` or `False`.')
    cell = electrode_pad(cell, electrode_pad_width, electrode_pad_height, pad_center1, layer_probe) # Add electrode pad on the right
    cell = electrode_pad(cell, electrode_pad_width, electrode_pad_height, pad_center2, layer_probe) # Add electrode pad on the left

    # Add tapered metal wire that extends the electrode pad and connects to the heater
    metal_wire_width1 = electrode_pad_width + electrode_gap + electrode_connect_width*2
    metal_wire_width2 = electrode_gap + electrode_connect_width*2
    metal_wire_height = electrode_pad_height/2 + electrode_pad_center[1] - 7/2*electrode_heater_width
    metal_wire_begin_point = ((pad_center1[0] + pad_center2[0])/2, pad_center1[1]+electrode_pad_height/2)
    cell = electrode_taper(
        cell, 
        metal_wire_height, 
        metal_wire_width1, 
        metal_wire_width2, 
        metal_wire_begin_point, 
        -np.pi/2, 
        layer_probe
    )

    # Add an auxiliary wire for booling operation that forms gap between cathode and anode electrodes
    metal_wire_height_auxiliary = electrode_pad_height/2 + electrode_pad_center[1] + electrode_heater_width*1/2
    metal_wire_width1_auxiliary = metal_wire_width2
    metal_wire_width2_auxiliary = electrode_gap
    cell = electrode_taper(
        cell, 
        metal_wire_height_auxiliary, 
        metal_wire_width1_auxiliary, 
        metal_wire_width2_auxiliary, 
        metal_wire_begin_point, 
        -np.pi/2, 
        layer_probe_auxiliary
    )

    # Add redundant heater to enhance the electrical contact between heater and electrode pad
    pad_heater_contact_width1 = electrode_pad_width/2 + electrode_gap + electrode_connect_width*2
    pad_heater_contact_width2 = electrode_gap + 2*electrode_heater_width
    pad_heater_contact_height = electrode_pad_height/2 + electrode_pad_center[1] - electrode_heater_width/2
    pad_heater_contact_begin_point = ((pad_center1[0] + pad_center2[0])/2, pad_center1[1]+electrode_pad_height/2)
    cell = electrode_taper(
        cell, 
        pad_heater_contact_height, 
        pad_heater_contact_width1, 
        pad_heater_contact_width2, 
        pad_heater_contact_begin_point, 
        -np.pi/2, 
        layer_heater
    )
    return cell

def waveguide_taper(
        cell, 
        taper_length, 
        taper_width1, 
        taper_width2, 
        etch_width, 
        angle, 
        begin_point, 
        layer_taper, 
        layer_wg_etch
    ):
    """
    Add a waveguide taper.

    Args:
        cell (gdspy.cell): A gdspy cell in which the waveguide taper will be added.
        taper_length (float): Length of the taper in um.
        taper_width1 (float): Width of the waveguide at the beginning of the taper in um.
        taper_width2 (float): Width of the waveguide at the end of the taper in um.
        etch_width (float): Width of the exposure regions in both sides of the waveguide in um.
        angle (float): Tilt angle of the taper in radian.
        begin_point (list): Position of the beginning point of the taper in um.
        layer_taper (int): Layer index of the taper.
        layer_wg_etch (int): Layer index of the etch regions around the taper.

    Returns:
        gdspy.cell: A gdspy cell with the waveguide taper.
        tuple: Position of the end point of the waveguide taper.
    """
    center = [0, 0] # Center of the taper
    center[0] = begin_point[0] + taper_length/2 * np.cos(angle)
    center[1] = begin_point[1] + taper_length/2 * np.sin(angle)
    taper_core_pts = [
        (-taper_length/2+center[0], -taper_width1/2+center[1]), 
        (taper_length/2+center[0], -taper_width2/2+center[1]), 
        (taper_length/2+center[0], taper_width2/2+center[1]), 
        (-taper_length/2+center[0], taper_width1/2+center[1]), 
        (-taper_length/2+center[0], -taper_width1/2+center[1])
    ] # Taper core vertices
    taper = gdspy.Polygon(taper_core_pts, layer=layer_taper) # taper core
    taper_clad_pts = [
        (-taper_length/2+center[0], -taper_width1/2-etch_width+center[1]), 
        (taper_length/2+center[0], -taper_width2/2-etch_width+center[1]), 
        (taper_length/2+center[0], taper_width2/2+etch_width+center[1]), 
        (-taper_length/2+center[0], taper_width1/2+etch_width+center[1]), 
        (-taper_length/2+center[0], -taper_width1/2-etch_width+center[1])
    ] # Taper clad vertices
    etch_taper = gdspy.Polygon(taper_clad_pts, layer=layer_wg_etch) # Waveguide cladding
    cell.add(taper.rotate(angle, center))
    cell.add(etch_taper.rotate(angle, center))

    # Calculate the end point of the taper
    end_point = geometry.rotate(center, begin_point, np.pi)
    return cell, end_point

def edge_etching(cell, pos1, pos2, layer_edge):
    """
    Add an edge coupler.

    Args:
        cell (gdspy.cell): A gdspy cell in which the edge coupler will be added.
        pos1 (tuple): Position of the lower left vertex of the edge region in um.
        pos2 (tuple): Position of the upper right vertex of the edge region in um.
        layer_taper (int): Layer index of the edge coupler.

    Returns:
        gdspy.cell: A gdspy cell with the waveguide.    
    
    Note:
        This function defines a region where deep etching will be applied.
    """
    edge = gdspy.Rectangle(pos1, pos2, layer=layer_edge)
    cell.add(edge)
    return cell

def dummy_marker(cell, region, marker_period, marker_width, marker_fineness, layer_marker):
    """
    Add cross-shaped dummy markers.

    Args:
        cell (gdspy.cell): A gdspy cell in which the dummy markers will be added.
        region (list): List of two points, specifying the region where dummy markers will be added.
        marker_period (float): Period of the periodic arrangement of dummy markers in um.
        marker_width (float): Width of the dummy marker in um.
        marker_fineness (int): Fineness of the dummy marker, i.e. width of each arm of the cross shape in um.
        layer_marker (int): Layer index of the dummy marker.
    
    Returns:
        gdspy.cell: A gdspy cell with the dummy markers.
    """
    # Determine position of markers
    marker_x = np.arange(region[0][0], region[1][0], marker_period)
    marker_y = np.arange(region[0][1], region[1][1], marker_period)
    marker_pos = []
    for x in marker_x:
        for y in marker_y:
            marker_pos.append((x, y))

    # Add markers
    for pos in marker_pos:
        marker = gdspy.Rectangle(
            (pos[0], pos[1]-marker_fineness/2+marker_width/2),
            (pos[0]+marker_width, pos[1]+marker_fineness/2+marker_width/2),
            layer=layer_marker
        )
        cell.add(marker)

        marker = gdspy.Rectangle(
            (pos[0]-marker_fineness/2+marker_width/2, pos[1]),
            (pos[0]+marker_fineness/2+marker_width/2, pos[1]+marker_width),
            layer=layer_marker
        )
        cell.add(marker)
    
    return cell

def process_testing_module(
        cell, 
        period_min, 
        period_step, 
        period_max, 
        feature_size_min, 
        feature_size_step, 
        feature_size_max, 
        sawtooth_height, 
        grating_width, 
        sample_distance, 
        text_height, 
        text_pos_bias, 
        testing_region_edge_width, 
        testing_region_size, 
        center, 
        number_of_points=64, 
        layer_wg=1, 
        layer_wg_etch=2, 
        layer_grating=3, 
        layer_text=4
    ):
    """
    Add gratings with various geometries and periods for process testing.
    
    Args:
        cell (gdspy.cell): A gdspy cell in which the gratings will be added.
        period_min (float): Minimum period of the grating in um.
        period_step (float): Step size of the period in um.
        period_max (float): Maximum period of the grating in um.
        feature_size_min (float): Minimum size of the grating feature in um.
        feature_size_step (float): Step size of the grating feature in um.
        feature_size_max (float): Maximum size of the grating feature in um.
        sawtooth_height (float): Height of the grating sawtooth in um.
        grating_width (float): Width of the zigzag grating in um, measured at the narrowest cross section.
        sample_distance (float): Distance between two neighboring samples in um.
        text_height (float): Height of the text in um.
        text_pos_bias (tuple): Bias of the text position in um.
        testing_region_edge_width (list): Width of the testing region boundary in the order of [left, right, bottom, top]
        testing_region_size (tuple): Size of the region where gratings will be added in um.
        center (tuple): Position of the center of the grating in um.
        number_of_points (int): Number of points used to approximate a silicon post.
        layer_wg (int): Layer index of the waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide.
        layer_grating (int): Layer index of the grating.
        layer_text (int): Layer index of the text.

    Returns:
        gdspy.cell: A gdspy cell with the gratings.
    """
    # Define the region where testing modules will be added
    testing_region = [
        (center[0]-testing_region_size[0]/2-testing_region_edge_width[0], center[1]-testing_region_size[1]/2-testing_region_edge_width[2]),
        (center[0]+testing_region_size[0]/2+testing_region_edge_width[1], center[1]+testing_region_size[1]/2+testing_region_edge_width[3])
    ]
    etching_region = gdspy.Rectangle(testing_region[0], testing_region[1], layer=layer_wg_etch)
    cell.add(etching_region)

    # Add waveguides with different widths
    idx_sample = 1
    width_list = np.arange(feature_size_min, feature_size_max, feature_size_step)
    for width in width_list:
        waveguide = gdspy.Rectangle(
            (testing_region[0][0]+testing_region_edge_width[0], testing_region[0][1]+idx_sample*sample_distance),
            (testing_region[1][0]-testing_region_edge_width[1], testing_region[0][1]+idx_sample*sample_distance+width),
            layer=layer_wg
        )
        cell.add(waveguide)
        # Add text label
        text_pos = [testing_region[1][0]+text_pos_bias[0], testing_region[0][1]+idx_sample*sample_distance+text_pos_bias[1]]
        text = gdspy.Text(str(idx_sample), text_height, text_pos, layer=layer_text)
        cell.add(text)
        idx_sample += 1

    # Add gratings with various geometries and periods
    period_list = np.arange(period_min, period_max, period_step)
    feature_size_list = np.arange(feature_size_min, feature_size_max, feature_size_step)
    
    # Add a post to form a post-array grating
    for feature_size in feature_size_list:
        idx_period = 0
        grating_x = -testing_region_size[0]/2 + period_max
        grating_y = -testing_region_size[1]/2 + feature_size/2 + sample_distance*idx_sample
        for period in period_list:
            single_post = gdspy.Round(center=(grating_x, grating_y), radius=feature_size/2, number_of_points=number_of_points, layer=layer_grating)
            cell.add(single_post.translate(center[0], center[1]))
            grating_x += period
            idx_period += 1
        # Add text label
        text_pos = [testing_region[1][0]+text_pos_bias[0], testing_region[0][1]+idx_sample*sample_distance+text_pos_bias[1]]
        text = gdspy.Text(str(idx_sample), text_height, text_pos, layer=layer_text)
        cell.add(text)
        idx_sample += 1

    # Add a sawtooth to form a zigzag grating
    for feature_size in feature_size_list:
        idx_period = 0
        grating_x = -testing_region_size[0]/2 + period_max
        grating_y += sample_distance
        for period in period_list:
            single_sawtooth = gdspy.Polygon(
                [
                    (grating_x-feature_size/2, grating_y),
                    (grating_x+feature_size/2, grating_y),
                    (grating_x+(period-feature_size)/2, grating_y+sawtooth_height),
                    (grating_x+period/2, grating_y+sawtooth_height),
                    (grating_x+period/2, grating_y+sawtooth_height+grating_width),
                    (grating_x-period/2, grating_y+sawtooth_height+grating_width),
                    (grating_x-period/2, grating_y+sawtooth_height),
                    (grating_x-(period-feature_size)/2, grating_y+sawtooth_height),
                ],
                layer=layer_grating
            )
            cell.add(single_sawtooth.translate(center[0], center[1]-(sawtooth_height+grating_width)/2))
            grating_x = grating_x + period + period_step/2
            idx_period += 1
        # Add text label
        text_pos = [testing_region[1][0]+text_pos_bias[0], testing_region[0][1]+idx_sample*sample_distance+text_pos_bias[1]]
        text = gdspy.Text(str(idx_sample), text_height, text_pos, layer=layer_text)
        cell.add(text)
        idx_sample += 1

    # Add a sinusoidal grating
    for feature_size in feature_size_list:
        idx_period = 0
        grating_x = -testing_region_size[0]/2 + period_min
        grating_y += sample_distance
        for period in period_list:
            sinusoidal_curve_x = np.linspace(0, period, number_of_points)
            sinusoidal_curve_y = feature_size/2 * np.cos(2*np.pi/period*sinusoidal_curve_x)
            sinusoidal_curve = np.stack((sinusoidal_curve_x, sinusoidal_curve_y), axis=1)
            sinusoidal_boundary = np.concatenate(
                (sinusoidal_curve, np.array([[sinusoidal_curve_x[-1], grating_width], [sinusoidal_curve_x[0], grating_width]])), 
                axis=0
            )
            single_sinusoidal = gdspy.Polygon(sinusoidal_boundary, layer=layer_grating)
            cell.add(single_sinusoidal.translate(center[0]+grating_x, center[1]+grating_y-(feature_size+grating_width)/2))
            grating_x = grating_x + period
            idx_period += 1
        # Add text label
        text_pos = [testing_region[1][0]+text_pos_bias[0], testing_region[0][1]+idx_sample*sample_distance+text_pos_bias[1]]
        text = gdspy.Text(str(idx_sample), text_height, text_pos, layer=layer_text)
        cell.add(text)
        idx_sample += 1

    return cell

def edge_coupler(
        cell,
        bus_wg_width,
        wg_etch_width,
        bus_wg_edgebend_radius,
        taper_angle,
        taper_length,
        taper_width,
        taper_marker=True,
        taper_marker_width_min=0.5,
        taper_marker_width_max=5,
        taper_marker_gap=20,
        begin_point=(0, 0),
        x_span=1000,
        orientation=0,
        layer_wg=1,
        layer_wg_etch=2
    ):
    """
    Add an edge coupler with an waveguide bend and a lateral taper marker.

    Args:
        cell (gdspy.cell): A gdspy cell in which the edge coupler will be added.
        bus_wg_width (float): Width of the bus waveguide in um.
        wg_etch_width (float): Width of the etch regions around the waveguide in um.
        bus_wg_edgebend_radius (float): Minimum curvature radius of the waveguide bend in um.
        taper_angle (float): Change of orientation angle across the waveguide bend.
        taper_length (float): Length of the taper in um.
        taper_width (float): Width of the taper measured on the side close to chip edge in um.
        taper_marker (bool): Whether to add a taper marker, default to `True`.
        taper_marker_width_min (float): Minimum width of the taper marker in um, default to `0.5`.
        taper_marker_width_max (float): Maximum width of the taper marker in um, default to `5`.
        taper_marker_gap (float): Separation distance between the taper and the taper marker in um, default to `20`.
        begin_point (tuple): Position of the beginning of the edge coupler in um, default to `(0, 0)`.
        x_span (float): Span of the edge coupler in x-direction in um, default to `1000`.
        orientation (float): Orientation angle of the edge coupler in radian, default to `0`.
        layer_wg (int): Layer index of the waveguide.
        layer_wg_etch (int): Layer index of the etch regions around the waveguide.
    
    """

    cell, port_pos = waveguide_bend_euler(
        cell, 
        bus_wg_width, 
        wg_etch_width, 
        bus_wg_edgebend_radius, 
        bend_angle=taper_angle, 
        bend_orientation=orientation, 
        begin_point=begin_point, 
        mirror_line=None, 
        number_of_points=64, 
        layer_bend=layer_wg, 
        layer_wg_etch=layer_wg_etch
    )
    wg_orientation = taper_angle + orientation

    cell, port_pos = waveguide_taper(
        cell, 
        taper_length, 
        bus_wg_width, 
        taper_width, 
        wg_etch_width, 
        wg_orientation, 
        port_pos, 
        layer_wg, 
        layer_wg_etch
    )

    extension_wg_length_correction = 5 # Increase the waveguide length to avoid gap between waveguide termination and chip edge

    cell = waveguide(
        cell, 
        (x_span - np.abs(port_pos[0] - begin_point[0])) / np.cos(taper_angle) + extension_wg_length_correction, 
        taper_width, 
        wg_etch_width, 
        wg_orientation, 
        port_pos, 
        layer_wg, 
        layer_wg_etch, 
        use_begin_point=True
    )

    if taper_marker:
        cell, _ = waveguide_taper(
            cell, 
            (x_span - np.abs(port_pos[0] - begin_point[0])) / np.cos(taper_angle), 
            taper_marker_width_min, 
            taper_marker_width_max, 
            wg_etch_width, 
            wg_orientation, 
            port_pos-np.array([0, taper_marker_gap]),
            layer_wg, 
            layer_wg_etch
        )

    return cell

def process_testing_module_metal(
        cell,
        periods,
        feature_size_max,
        feature_size_min,
        feature_size_step,
        disk_outer_radius,
        etch_metal_gap,
        begin_point,
        text_height,
        layer_wg_etch,
        layer_heater,
        layer_probe,
        layer_text,
        layer_text_frame
    ):
    """
    Add some features for metal lifting process and maskless lithography testing.

    Args:
        cell (gdspy.cell): A gdspy cell in which the features will be added.
        periods (tuple): Periods of the features in x and y directions in um.
        feature_size_max (float): Maximum size of the feature in um.
        feature_size_min (float): Minimum size of the feature in um.
        feature_size_step (float): Step size of the feature in um.
        disk_outer_radius (float): Outer radius of disk-shaped feater in um.
        etch_metal_gap (float): Gap between the metal and the etch region boundary in um.
        begin_point (tuple): Position of the beginning of the features in um.
        text_height (float): Height of text label in um.
        layer_wg_etch (int): Layer index of the etch regions in the waveguide layer.
        layer_heater (int): Layer index of the heater.
        layer_probe (int): Layer index of the probe for electrical contact.
        layer_text (int): Layer index of the text.
        layer_text_frame (int): Layer index of the text frame.

    Returns:
        gdspy.cell: A gdspy cell with the features.
    """
    # Define periodic array of features
    feature_num = round((feature_size_max - feature_size_min) / feature_size_step) + 1

    # Define size of text labels
    text_distance = 8 / 9 * text_height

    # Add array of heater metal posts
    for idx_post in range(feature_num):
        feature_center = np.array(begin_point) + np.array([idx_post * periods[0], 0])
        post_diameter = feature_size_min + idx_post * feature_size_step
        metal_post = gdspy.Round(feature_center, post_diameter/2, layer=layer_heater, number_of_points=128)
        cell.add(metal_post)
        etch_diameter = post_diameter + 2 * etch_metal_gap
        etch_post = gdspy.Round(feature_center, etch_diameter/2, layer=layer_wg_etch, number_of_points=128)
        cell.add(etch_post)

    # Add text label for the first row
    text_pos = np.array(begin_point) + np.array([0, -periods[1] / 2 - text_height / 2])
    label = "post d=" + "{:.2f}".format(feature_size_min) + " to {:.2f}".format(feature_size_max)
    text_label = gdspy.Text(label, text_height, text_pos, layer=layer_text)
    cell.add(text_label)
    frame_size = np.array([text_distance * len(label), text_height])
    text_frame = gdspy.Rectangle(
        text_pos + np.array([-2/9 * text_distance, -2/9 * text_height]), 
        text_pos + np.array([frame_size[0] + 2/9 * text_distance, 11 / 9 * text_height]), 
        layer=layer_text_frame
    )
    cell.add(text_frame)

    # Add array of probe metal posts
    for idx_post in range(feature_num):
        feature_center = np.array(begin_point) + np.array([idx_post * periods[0], -periods[1]])
        post_diameter = feature_size_min + idx_post * feature_size_step
        metal_post = gdspy.Round(feature_center, post_diameter/2, layer=layer_probe, number_of_points=128)
        cell.add(metal_post)
        etch_diameter = post_diameter + 2 * etch_metal_gap
        etch_post = gdspy.Round(feature_center, etch_diameter/2, layer=layer_wg_etch, number_of_points=128)
        cell.add(etch_post)

    # Add text label for the second row
    text_pos = np.array(begin_point) + np.array([0, -3 * periods[1] / 2 - text_height / 2])
    label = "post d=" + "{:.2f}".format(feature_size_min) + " to {:.2f}".format(feature_size_max)
    text_label = gdspy.Text(label, text_height, text_pos, layer=layer_text)
    cell.add(text_label)
    frame_size = np.array([text_distance * len(label), text_height])
    text_frame = gdspy.Rectangle(
        text_pos + np.array([-2/9 * text_distance, -2/9 * text_height]), 
        text_pos + np.array([frame_size[0] + 2/9 * text_distance, 11 / 9 * text_height]), 
        layer=layer_text_frame
    )
    cell.add(text_frame)

    # Add array of heater metal disks with holes
    for idx_disk in range(feature_num):
        feature_center = np.array(begin_point) + np.array([idx_disk * periods[0], -2 * periods[1]])
        metal_disk = gdspy.Round(feature_center, disk_outer_radius/2, layer=layer_heater, number_of_points=128)
        disk_hole_diameter = feature_size_min + feature_size_step * idx_disk
        metal_disk_hole = gdspy.Round(feature_center, disk_hole_diameter/2, layer=layer_heater, number_of_points=128)
        metal_disk = gdspy.boolean(metal_disk, metal_disk_hole, 'not', layer=layer_heater)
        cell.add(metal_disk)
        etch_diameter = disk_outer_radius + 2 * etch_metal_gap
        etch_disk = gdspy.Round(feature_center, etch_diameter/2, layer=layer_wg_etch, number_of_points=128)
        cell.add(etch_disk)

    # Add text label for the third row
    text_pos = np.array(begin_point) + np.array([0, -5 * periods[1] / 2 - text_height / 2])
    label = "disk d=" + "{:.2f}".format(feature_size_min) + " to {:.2f}".format(feature_size_max)
    text_label = gdspy.Text(label, text_height, text_pos, layer=layer_text)
    cell.add(text_label)
    frame_size = np.array([text_distance * len(label), text_height])
    text_frame = gdspy.Rectangle(
        text_pos + np.array([-2/9 * text_distance, -2/9 * text_height]), 
        text_pos + np.array([frame_size[0] + 2/9 * text_distance, 11 / 9 * text_height]), 
        layer=layer_text_frame
    )
    cell.add(text_frame)
        
    # Add array of probe metal disks with holes
    for idx_disk in range(feature_num):
        feature_center = np.array(begin_point) + np.array([idx_disk * periods[0], -3 * periods[1]])
        metal_disk = gdspy.Round(feature_center, disk_outer_radius/2, layer=layer_probe, number_of_points=128)
        disk_hole_diameter = feature_size_min + feature_size_step * idx_disk
        metal_disk_hole = gdspy.Round(feature_center, disk_hole_diameter/2, layer=layer_probe, number_of_points=128)
        metal_disk = gdspy.boolean(metal_disk, metal_disk_hole, 'not', layer=layer_probe)
        cell.add(metal_disk)
        etch_diameter = disk_outer_radius + 2 * etch_metal_gap
        etch_disk = gdspy.Round(feature_center, etch_diameter/2, layer=layer_wg_etch, number_of_points=128)
        cell.add(etch_disk)

    # Add text label for the fourth row
    text_pos = np.array(begin_point) + np.array([0, -7 * periods[1] / 2 - text_height / 2])
    label = "disk d=" + "{:.2f}".format(feature_size_min) + " to {:.2f}".format(feature_size_max)
    text_label = gdspy.Text(label, text_height, text_pos, layer=layer_text)
    cell.add(text_label)
    frame_size = np.array([text_distance * len(label), text_height])
    text_frame = gdspy.Rectangle(
        text_pos + np.array([-2/9 * text_distance, -2/9 * text_height]), 
        text_pos + np.array([frame_size[0] + 2/9 * text_distance, 11 / 9 * text_height]), 
        layer=layer_text_frame
    )
    cell.add(text_frame)

    return cell

def alignment_marker(
        cell,
        marker_pos,
        layer_alignment_marker,
        layer_alignment_marker_frame,
        layer_overlay_1,
        layer_overlay_2
    ):
    """
    Add alignment markers and Vernier scales for mask alignment.

    Args:
        cell (gdspy.cell): A gdspy cell in which the alignment markers will be added.
        marker_pos (list): List of positions of the alignment markers in um.
        layer_alignment_marker (int): Layer index of the alignment markers.
        layer_alignment_marker_frame (int): Layer index of the rectangular frame of the alignment marker.
        layer_overlay_1 (int): Layer index of the first overlay on the waveguide layer.
        layer_overlay_2 (int): Layer index of the second overlay on the waveguide layer.

    Returns:
        gdspy.cell: A gdspy cell with the alignment markers.
    """
    # Parameters of alignment marker
    marker_size = [475, 475]
    marker_cross_inner_diameter = 5
    marker_cross_inner_width = 1
    marker_cross_outer_diameter = 100
    marker_cross_outer_width = 10
    marker_cross_large_width = 200
    marker_cross_large_height = 400
    marker_cross_large_thick = 10
    marker_cross_small_width = 60
    marker_cross_small_height = 60
    marker_corss_small_thick = 10
    marker_bar_width = 10
    marker_bar_length = 190
    marker_bar_gap = 10
    marker_bar_num = 10 # Number of bar-shaped alignment markers in each quadrant
    marker_bar_offset = [10, 10]
    marker_sep = marker_size[0]

    # Parameters of Vernier scale
    Vernier_square_outer_size = [190, 190]
    Vernier_square_outer_thick = 10
    Vernier_square_inner_size = [130, 130]
    Vernier_square_inner_thick = 10
    Vernier_bar_num = 21
    Vernier_bar_width = 5
    Vernier_bar_length = 30
    Vernier_bar_pitch = [5.1, 5, 5.5]
    Vernier_ref_bar_length = [10, 20]
    Vernier_ref_bar_num = round((Vernier_bar_num - 1) / 5) + 1
    Vernier_ref_bar_period = 50

    # Add alignment marker
    for pos in marker_pos:
        pos = np.array(pos) + np.array([marker_size[0]/2, marker_size[1]/2])

        # Add frame for alignment marker
        marker_num = 5
        frame = gdspy.Rectangle(
            pos - np.array(marker_size) / 2,
            [pos[0] + np.array(marker_size[0]) / 2 * (2 * marker_num - 1), pos[1] + np.array(marker_size[1])/2],
            layer=layer_alignment_marker_frame
        )
        cell.add(frame)

        # Add cross-shaped alignment marker
        cross_inner_bar1 = gdspy.Rectangle(
            pos - np.array([marker_cross_inner_diameter/2, marker_cross_inner_width/2]),
            pos + np.array([marker_cross_inner_diameter/2, marker_cross_inner_width/2]),
        )
        cross_inner_bar2 = gdspy.Rectangle(
            pos - np.array([marker_cross_inner_width/2, marker_cross_inner_diameter/2]),
            pos + np.array([marker_cross_inner_width/2, marker_cross_inner_diameter/2]),
        )
        cross_inner = gdspy.boolean(cross_inner_bar1, cross_inner_bar2, 'or')
        cross_outer_bar1 = gdspy.Rectangle(
            pos - np.array([marker_cross_outer_diameter/2, marker_cross_outer_width/2]),
            pos + np.array([marker_cross_outer_diameter/2, marker_cross_outer_width/2]),
        )
        cross_outer_bar2 = gdspy.Rectangle(
            pos - np.array([marker_cross_outer_width/2, marker_cross_outer_diameter/2]),
            pos + np.array([marker_cross_outer_width/2, marker_cross_outer_diameter/2]),
        )
        cross_outer = gdspy.boolean(cross_outer_bar1, cross_outer_bar2, 'or')
        cross_marker = gdspy.boolean(cross_outer, cross_inner, 'not', layer=layer_alignment_marker)
        cell.add(cross_marker)

        # Add bar-shaped alignment marker
        for idx_quadrant in range(4):
            for idx_bar in range(marker_bar_num):
                bar_pos = pos + np.array(marker_bar_offset) + np.array([0, idx_bar * (marker_bar_width + marker_bar_gap)])
                bar = gdspy.Rectangle(
                    bar_pos,
                    bar_pos + np.array([marker_bar_length, marker_bar_width]),
                    layer=layer_alignment_marker
                ).rotate(np.pi/2 * idx_quadrant, pos)
                cell.add(bar)

        # Add single big cross-shaped alignment marker
        pos = pos + np.array([marker_sep, 0])
        cross_large_bar1 = gdspy.Rectangle(
            pos - np.array([marker_cross_large_thick/2, marker_cross_large_height/2]),
            pos + np.array([marker_cross_large_thick/2, marker_cross_large_height/2]),
        )
        cross_large_bar2 = gdspy.Rectangle(
            pos - np.array([marker_cross_large_width/2, marker_cross_large_thick/2]),
            pos + np.array([marker_cross_large_width/2, marker_cross_large_thick/2]),
        )
        cross_large = gdspy.boolean(cross_large_bar1, cross_large_bar2, 'or', layer=layer_alignment_marker)
        cell.add(cross_large)
        cross_large = gdspy.boolean(cross_large_bar1, cross_large_bar2, 'or', layer=layer_overlay_1)
        cell.add(cross_large)
        cross_large = gdspy.boolean(cross_large_bar1, cross_large_bar2, 'or', layer=layer_overlay_2)
        cell.add(cross_large)

        # Add single small cross-shaped alignment marker
        pos = pos + np.array([marker_sep, 0])
        cross_small_bar1 = gdspy.Rectangle(
            pos - np.array([marker_corss_small_thick/2, marker_cross_small_height/2]),
            pos + np.array([marker_corss_small_thick/2, marker_cross_small_height/2]),
        )
        cross_small_bar2 = gdspy.Rectangle(
            pos - np.array([marker_cross_small_width/2, marker_corss_small_thick/2]),
            pos + np.array([marker_cross_small_width/2, marker_corss_small_thick/2]),
        )
        cross_small = gdspy.boolean(cross_small_bar1, cross_small_bar2, 'or', layer=layer_alignment_marker)
        cell.add(cross_small)
        cross_small = gdspy.boolean(cross_small_bar1, cross_small_bar2, 'or', layer=layer_overlay_1)
        cell.add(cross_small)
        cross_small = gdspy.boolean(cross_small_bar1, cross_small_bar2, 'or', layer=layer_overlay_2)
        cell.add(cross_small)

        # Add Vernier scale to check alignment between masks of different layers
        for idx_overlay_layer in [layer_overlay_1, layer_overlay_2]:
            pos = pos + np.array([marker_sep, 0])
            # Add outer square marker
            Vernier_square_outer = gdspy.Rectangle(pos-np.array(Vernier_square_outer_size), pos)
            Vernier_square_hole = gdspy.Rectangle(
                pos-np.array(Vernier_square_outer_size)+np.array([Vernier_square_outer_thick, Vernier_square_outer_thick]), 
                pos-np.array([Vernier_square_outer_thick, Vernier_square_outer_thick]), 
            )
            Vernier_square_outer = gdspy.boolean(Vernier_square_outer, Vernier_square_hole, 'not', layer=layer_alignment_marker)
            cell.add(Vernier_square_outer)
            # Add inner square marker
            square_marker_center = pos - np.array(Vernier_square_outer_size) / 2
            Vernier_square_inner = gdspy.Rectangle(
                square_marker_center - np.array(Vernier_square_inner_size) / 2,
                square_marker_center + np.array(Vernier_square_inner_size) / 2,
            )
            Vernier_square_hole = gdspy.Rectangle(
                square_marker_center - np.array(Vernier_square_inner_size) / 2 + np.array([Vernier_square_inner_thick, Vernier_square_inner_thick]),
                square_marker_center + np.array(Vernier_square_inner_size) / 2 - np.array([Vernier_square_inner_thick, Vernier_square_inner_thick]),
            )
            Vernier_square_inner = gdspy.boolean(Vernier_square_inner, Vernier_square_hole, 'not', layer=idx_overlay_layer)
            cell.add(Vernier_square_inner)
            
            for idx_quadrant in range(2):    
                # Add Vernier scale
                # Vernier_scale_center = pos + np.array([-marker_size[0] / 4, marker_size[1] / 4])
                Vernier_scale_center = square_marker_center + np.array([0, Vernier_square_outer_size[1]])
                for idx_row in range(len(Vernier_bar_pitch)):
                    for idx_col in range(Vernier_bar_num):
                        bar_pos = Vernier_scale_center + np.array([idx_col * (Vernier_bar_pitch[idx_row]+Vernier_bar_width), idx_row * Vernier_bar_length])
                        bar_pos[0] = bar_pos[0] - (Vernier_bar_num - 1) / 2 * (Vernier_bar_width + Vernier_bar_pitch[idx_row]) - Vernier_bar_width / 2
                        bar_pos[1] = bar_pos[1] - Vernier_bar_length / 2 * len(Vernier_bar_pitch)
                        Vernier_scale_layer = idx_overlay_layer if idx_row % 2 == 0 else layer_alignment_marker
                        bar = gdspy.Rectangle(
                            bar_pos,
                            bar_pos + np.array([Vernier_bar_width, Vernier_bar_length]),
                            layer=Vernier_scale_layer
                        )
                        cell.add(bar.rotate(-np.pi/2 * idx_quadrant, square_marker_center))
                # Add reference bar
                for idx_row in range(2):
                    for idx_col in range(Vernier_ref_bar_num):
                        bar_pos = Vernier_scale_center + np.array([idx_col * Vernier_ref_bar_period, idx_row * Vernier_bar_length * len(Vernier_bar_pitch)])
                        bar_pos[0] = bar_pos[0] - Vernier_ref_bar_period * (Vernier_ref_bar_num - 1) / 2- Vernier_bar_width / 2
                        bar_pos[1] = bar_pos[1] - Vernier_bar_length * len(Vernier_bar_pitch) / 2
                        bar_length = Vernier_ref_bar_length[0] if idx_col % 2 == 0 else Vernier_ref_bar_length[1]
                        bar_length = bar_length if idx_row == 1 else -bar_length
                        bar = gdspy.Rectangle(
                            bar_pos,
                            bar_pos + np.array([Vernier_bar_width, bar_length]),
                            layer=layer_alignment_marker
                        )
                        cell.add(bar.rotate(-np.pi/2 * idx_quadrant, square_marker_center))
            
    return cell

def dicing_line(
        cell,
        pos1,
        pos2,
        layer_dicing_lines
    ):
    """
    Add dicing lines for chip dicing.

    Args:
        cell (gdspy.cell): A gdspy cell in which the dicing lines will be added.
        pos1 (tuple): Position of the first end point of the dicing line in um.
        pos2 (tuple): Position of the second end point of the dicing line in um.
        layer_dicing_lines (int): Layer index of the dicing lines.

    Returns:
        gdspy.cell: A gdspy cell with the dicing lines.
    """
    dicing_width = 0.001
    dicing_line = gdspy.Rectangle(
        np.array(pos1) - np.array([0, dicing_width/2]),
        np.array(pos2) + np.array([0, dicing_width/2]),
        layer=layer_dicing_lines
    )
    cell.add(dicing_line)
    return cell