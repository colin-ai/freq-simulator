import warnings

from numpy import abs, arctan, argmax, linspace, min, sign, sin


def magic_formula(p, num=1000000, b=10, c=1.9, d=1, e=0.97):
    """

    The magic formula [1] computes the longitudinal force between wheel and tarmac
    in function of the longitudinal tire slip.

    [1] Pacejka, H. B. Tire and Vehicle Dynamics. Elsevier Science, 2005

    Parameters
    ----------
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    p : float
        Weight applied on the tire [N].
    b : float
        Stiffness factor. Default value corresponds to dry tarmac.
    c : float
        Shape factor. Default value corresponds to dry tarmac.
    d : float
        Peak factor. Default value corresponds to dry tarmac.
    e : float
        Curvature factor. Default value corresponds to dry tarmac.

    Returns
    -------
    F : float or array
        If x is a slip ratio, Func is the longitudinal force between tire and tarmac [N], if x is a wheel slip angle, Func
        is the lateral force.
    alpha : float, array_like
        Tire slip ratio or slip angle.
    """
    alpha = linspace(
        0, 0.99, num
    )  # Space limited to 0.99 to avoid locked sliding wheel at x = 1
    F = p * d * sin(c * arctan(b * alpha * (1 - e) + e * arctan(b * alpha)))

    return F, alpha


def curve_switch_value(array):
    """
    It computes argmax, max and of an array and its min value on the right part of the curve.
    """
    switch_idx = argmax(array)
    max_value = array[switch_idx]
    min_value_right = min(array[switch_idx + 1 :])
    return switch_idx, max_value, min_value_right


def switcher(switch_value, v0, v1, switch_state, verbose):
    """
    It updates the switch state according inp values, either "left" of the switch value or "right",
    respectively swith_state = 0 and switch_state = 1.
    """
    if abs(v0) < switch_value < abs(v1):
        switch_state = (switch_state + 1) % 2
        if verbose:
            print("Switch point crossed (bottom > up), switch_state =", switch_state)
    if abs(v0) > switch_value > abs(v1):
        switch_state = (switch_state + 1) % 2
        if verbose:
            print("Switch point crossed (up > bottom), switch_state =", switch_state)
    return switch_state


def curve_area(x_array, y_array, switch_idx, switch_state):
    """
    It truncates the curve space according the current switch_state.
    """
    if switch_state == 0:
        x_array_reduced = x_array[:switch_idx]
        y_array_reduced = y_array[:switch_idx]
    else:
        x_array_reduced = x_array[switch_idx + 1 :]
        y_array_reduced = y_array[switch_idx + 1 :]
    return x_array_reduced, y_array_reduced


def force_corrector(F_value, F_min, F_max):
    """
    It rectifies the longitudinal force Func when it is higher than maximum value F_max given by Magic formula.
    In physic terms, Func is no more equal to m*a.
    """

    F_real = 2 * F_max * sign(F_value) - F_value

    if (F_value > 0) & (F_real < F_min):
        F_real = F_min
        warnings.warn(
            "Acceleration leads to slip value > 0.99. "
            "Propulsion has been capped to obtain a slip value = 0.99."
        )

    if (F_value < 0) & (F_real > -F_min):
        F_real = -F_min
        warnings.warn(
            "Deceleration leads to slip value < 0.99. "
            "Propulsion has been capped to obtain a slip value of -0.99."
        )

    return F_real


def slip_ratio(array_src, array_target, value):
    """
    if V_vehicle > V_roue :
        slip_ratio = (V_vehicle - V_wheel) / V_vehicle
    if V_vehicle < V_roue :
        slip_ratio = (V_wheel - V_vehicle) / V_wheel

    Source : Comprehensive prediction method of road friction for vehicle dynamics control

    """
    value = abs(value)
    idx = abs(array_src - value).argmin()
    value_target = array_target[idx]
    return value_target


def wheel_slip_velocity(v_vehicle, F, slip_rate):
    """
    It computes wheel slip velocity from slip_rate and propulsion Func.
    """
    v_wheel = v_vehicle
    if F >= 0:
        v_wheel = v_vehicle / (slip_rate + 1)
    if F < 0:
        v_wheel = v_vehicle * (slip_rate + 1)
    return v_wheel


def slip_velocities(v_vehicle, m, freq, F, k, verbose=False):
    """
    Given an array of vehicle position, it computes an array of wheel slip position of same lenght.

    Parameters
    ----------
    v_vehicle : array_like
        Input vehicle position in [m/s].
    m : float
        Mass of vehicle [kg].
    freq : int
        Recording frequency or number of frame per second.
    F : array_like
        Propulsion given by Magic formula [1].
    k : array_like
        Slip ratio used as inp to compute propulsion Func [1].

    Returns
    -------
    v_wheel_list : array
        Wheel slip position.
    Fx_list : array
        Propulsion.
    slips : array
        Slip ratios.

    """

    F_max_idx, F_max, F_min = curve_switch_value(F)

    v_wheel_list = [v_vehicle[0]]
    slips = []
    Fx_list = []
    N = len(v_vehicle)
    switch_state = 0
    F0 = 0

    for i in range(N - 1):
        if verbose:
            print(i, "/", N - 2)

        v0 = v_vehicle[i]
        v1 = v_vehicle[i + 1]

        F1 = m * (v1 - v0) / freq
        switch_state = switcher(
            F_max, F0, F1, switch_state, verbose
        )  # update state if F_max is reached
        F0 = F1
        k_area, F_area = curve_area(k, F, F_max_idx, switch_state)
        if switch_state == 1:
            F1 = force_corrector(F1, F_min, F_max)
        slip = slip_ratio(F_area, k_area, F1)
        v_wheel = wheel_slip_velocity(v1, F1, slip)

        v_wheel_list.append(v_wheel)
        Fx_list.append(F1)
        slips.append(slip)

    return v_wheel_list, Fx_list, slips
