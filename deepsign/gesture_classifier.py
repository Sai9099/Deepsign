"""
Comprehensive Rule-based ASL Static Gesture Classifier.

Uses hand landmark geometry (finger states, angles, distances) to classify
all 26 static ASL alphabet signs. Works without any trained ML model.

MediaPipe hand landmark indices:
  0: WRIST
  1-4: THUMB (CMC, MCP, IP, TIP)
  5-8: INDEX (MCP, PIP, DIP, TIP)
  9-12: MIDDLE (MCP, PIP, DIP, TIP)
  13-16: RING (MCP, PIP, DIP, TIP)
  17-20: PINKY (MCP, PIP, DIP, TIP)

Note: J and Z are motion-based signs in ASL. This classifier approximates
them using static hand shapes (J ~ pinky extended with hand angled,
Z ~ index pointing with hand angled).
"""

import math


# ─── Geometry Helpers ───────────────────────────────────────────────

def _dist(a, b):
    """Euclidean distance between two 3D landmarks."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def _dist2d(a, b):
    """Euclidean distance between two 2D landmarks (ignoring z)."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def _angle_at(a, b, c):
    """Angle in degrees at point b formed by line segments b->a and b->c."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = sum(i * j for i, j in zip(ba, bc))
    mag_ba = math.sqrt(sum(v**2 for v in ba))
    mag_bc = math.sqrt(sum(v**2 for v in bc))
    if mag_ba * mag_bc == 0:
        return 0
    cos_a = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))


# ─── Finger State Analysis ─────────────────────────────────────────

def _is_finger_extended(lms, finger):
    """
    Robust finger extension check using multiple criteria.
    Returns True if the finger is extended (straight/pointing out).
    """
    idx = {
        'thumb':  (1, 2, 3, 4),
        'index':  (5, 6, 7, 8),
        'middle': (9, 10, 11, 12),
        'ring':   (13, 14, 15, 16),
        'pinky':  (17, 18, 19, 20),
    }
    mcp_i, pip_i, dip_i, tip_i = idx[finger]
    mcp, pip, dip, tip = lms[mcp_i], lms[pip_i], lms[dip_i], lms[tip_i]
    wrist = lms[0]

    if finger == 'thumb':
        index_mcp = lms[5]
        tip_to_idx = _dist2d(tip, index_mcp)
        ip_to_idx = _dist2d(pip, index_mcp)
        tip_wrist = _dist2d(tip, wrist)
        mcp_wrist = _dist2d(mcp, wrist)
        return tip_to_idx > ip_to_idx * 1.05 and tip_wrist > mcp_wrist * 0.9
    else:
        # Distance-based check (orientation independent)
        tip_dist = _dist2d(tip, wrist)
        pip_dist = _dist2d(pip, wrist)
        mcp_dist = _dist2d(mcp, wrist)
        
        # Angle check (is the finger straight?)
        pip_angle = _angle_at(mcp, pip, dip)
        is_straight = pip_angle > 145
        
        # A finger is extended if it's pointing away from the wrist
        # and doesn't have a significant bend.
        is_pointing_away = tip_dist > pip_dist * 1.1 and tip_dist > mcp_dist * 1.2
        
        score = int(is_straight) + int(is_pointing_away)
        
        # We also check if the tip is significantly further from the wrist than the MCP
        # This helps with hand angles where the finger is "pointing at the camera"
        tip_far = tip_dist > mcp_dist * 1.3
        
        return score >= 2 or (is_straight and tip_far)


def _is_finger_curled(lms, finger):
    """Check if a finger is significantly curled (PIP angle < 130)."""
    if finger == 'thumb':
        return _angle_at(lms[1], lms[2], lms[3]) < 130
    idx = {
        'index': (5, 6, 7),
        'middle': (9, 10, 11),
        'ring': (13, 14, 15),
        'pinky': (17, 18, 19),
    }
    mcp_i, pip_i, dip_i = idx[finger]
    return _angle_at(lms[mcp_i], lms[pip_i], lms[dip_i]) < 130


def _is_finger_partially_curled(lms, finger):
    """Check if a finger is partially curled (not straight, not fully closed).
    PIP angle between 90 and 160 degrees."""
    if finger == 'thumb':
        angle = _angle_at(lms[1], lms[2], lms[3])
        return 70 < angle < 160
    idx = {
        'index': (5, 6, 7),
        'middle': (9, 10, 11),
        'ring': (13, 14, 15),
        'pinky': (17, 18, 19),
    }
    mcp_i, pip_i, dip_i = idx[finger]
    angle = _angle_at(lms[mcp_i], lms[pip_i], lms[dip_i])
    return 70 < angle < 165


def _get_finger_states(lms):
    """Returns dict of which fingers are extended."""
    return {
        'thumb':  _is_finger_extended(lms, 'thumb'),
        'index':  _is_finger_extended(lms, 'index'),
        'middle': _is_finger_extended(lms, 'middle'),
        'ring':   _is_finger_extended(lms, 'ring'),
        'pinky':  _is_finger_extended(lms, 'pinky'),
    }


def _count_extended(fs):
    return sum(fs.values())


def _tips_close(lms, i, j, threshold=0.06):
    """Check if two landmark points are close together."""
    return _dist(lms[i], lms[j]) < threshold


def _finger_curl_angle(lms, finger):
    """Get the curl angle of a finger (angle at PIP joint)."""
    idx = {
        'index': (5, 6, 7),
        'middle': (9, 10, 11),
        'ring': (13, 14, 15),
        'pinky': (17, 18, 19),
    }
    if finger == 'thumb':
        return _angle_at(lms[1], lms[2], lms[3])
    mcp_i, pip_i, dip_i = idx[finger]
    return _angle_at(lms[mcp_i], lms[pip_i], lms[dip_i])


def _fingers_spread(lms):
    """Check if extended fingers are spread apart."""
    tips = [8, 12, 16, 20]
    total_dist = 0
    count = 0
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            total_dist += _dist2d(lms[tips[i]], lms[tips[j]])
            count += 1
    return (total_dist / count) > 0.12 if count > 0 else False


def _index_pointing_sideways(lms):
    """Check if index finger is pointing horizontally."""
    dx = abs(lms[8].x - lms[5].x)
    dy = abs(lms[8].y - lms[5].y)
    return dx > dy * 1.2


def _hand_orientation(lms):
    """Get rough hand orientation: 'up', 'down', 'side'."""
    wrist = lms[0]
    middle_mcp = lms[9]
    dx = abs(middle_mcp.x - wrist.x)
    dy = middle_mcp.y - wrist.y  # negative = fingers up

    if dy < -0.05 and abs(dy) > dx:
        return 'up'
    elif dy > 0.05 and abs(dy) > dx:
        return 'down'
    else:
        return 'side'


def _palm_size(lms):
    """Approximate palm size as distance from wrist to middle MCP."""
    return _dist2d(lms[0], lms[9])


# ─── Shape-Based Detectors (bypass extension counting) ─────────────

def _detect_C_shape(lms):
    """
    C: All fingers curved together, thumb opposed, forming a C opening.
    Key: significant gap between thumb tip and index tip, all fingers
    partially curled in the same direction, fingers together.
    """
    palm = _palm_size(lms)
    if palm < 0.01:
        return False, 0.0

    thumb_tip = lms[4]
    index_tip = lms[8]

    # Gap between thumb and index (normalized by palm size)
    gap = _dist2d(thumb_tip, index_tip) / palm
    if gap < 0.15 or gap > 1.5:
        return False, 0.0

    # Fingers should be partially curled (not straight, not fully closed)
    partially_curled_count = 0
    for finger in ['index', 'middle', 'ring', 'pinky']:
        if _is_finger_partially_curled(lms, finger):
            partially_curled_count += 1
        else:
            # Also count fingers with moderate curl angles as "partially curled"
            angle = _finger_curl_angle(lms, finger)
            if 60 < angle < 170:
                partially_curled_count += 1

    if partially_curled_count < 1:
        return False, 0.0

    # Fingers should be roughly together (tips not too spread)
    finger_tips = [lms[8], lms[12], lms[16], lms[20]]
    max_spread = 0
    for i in range(len(finger_tips)):
        for j in range(i + 1, len(finger_tips)):
            d = _dist2d(finger_tips[i], finger_tips[j])
            max_spread = max(max_spread, d)

    # Thumb should be roughly on the opposite side from fingers (forming opening)
    # Use generous tolerance for thumb position relative to finger range
    min_y = min(index_tip.y, lms[20].y)
    max_y = max(index_tip.y, lms[20].y)
    thumb_between = (min_y - 0.12 <= thumb_tip.y <= max_y + 0.12)

    # Also check: thumb tip should be separated from index tip horizontally
    thumb_separated = abs(thumb_tip.x - index_tip.x) > 0.02 or gap > 0.3

    if partially_curled_count >= 3 and (thumb_between or thumb_separated):
        conf = 0.88 + (0.04 if partially_curled_count == 4 else 0)
        return True, min(conf, 0.93)

    if partially_curled_count >= 2 and (thumb_between or thumb_separated):
        return True, 0.82

    if partially_curled_count >= 1 and gap > 0.25:
        return True, 0.75

    return False, 0.0


def _detect_O_shape(lms):
    """
    O: Thumb tip and index tip touching or nearly touching, forming a circle.
    Other fingers also curve inward to complete the O shape.
    """
    palm = _palm_size(lms)
    if palm < 0.01:
        return False, 0.0

    # Thumb and index tips must be close
    thumb_index_dist = _dist(lms[4], lms[8]) / palm
    if thumb_index_dist > 0.5:
        return False, 0.0

    # Other fingers should be somewhat curled
    mid_curl = _finger_curl_angle(lms, 'middle')
    ring_curl = _finger_curl_angle(lms, 'ring')

    curled_count = 0
    if mid_curl < 160:
        curled_count += 1
    if ring_curl < 160:
        curled_count += 1
    if _finger_curl_angle(lms, 'pinky') < 160:
        curled_count += 1

    if thumb_index_dist < 0.25 and curled_count >= 1:
        conf = 0.88 if thumb_index_dist < 0.15 else 0.80
        return True, conf

    if thumb_index_dist < 0.35:
        return True, 0.72

    return False, 0.0


def _detect_F_shape(lms, fs):
    """
    F: Thumb and index form a circle (touching), middle+ring+pinky extended up.
    Like an OK sign.
    """
    palm = _palm_size(lms)
    if palm < 0.01:
        return False, 0.0

    # Thumb and index tips must be close (forming circle)
    thumb_index_dist = _dist(lms[4], lms[8]) / palm
    if thumb_index_dist > 0.55:
        return False, 0.0

    # Middle, ring, pinky should be extended
    extended = 0
    if fs['middle']:
        extended += 1
    if fs['ring']:
        extended += 1
    if fs['pinky']:
        extended += 1

    if extended >= 2 and thumb_index_dist < 0.4:
        conf = 0.88 if extended == 3 else 0.80
        return True, conf

    return False, 0.0


# ─── Main Classifier ───────────────────────────────────────────────

def classify_gesture(hand_landmarks):
    """
    Classify an ASL static hand gesture from MediaPipe NormalizedLandmarks.

    Args:
        hand_landmarks: list of 21 NormalizedLandmark objects

    Returns:
        (letter, accuracy, debug_info) tuple. 
        letter is a string A-Z or None.
        accuracy is a float 0.0-1.0.
        debug_info is a dict with 'ext' and 'orient'.
    """
    lms = hand_landmarks
    if len(lms) != 21:
        return None, 0.0, {'ext': 0, 'orient': 'none'}

    fs = _get_finger_states(lms)
    thumb  = fs['thumb']
    index  = fs['index']
    middle = fs['middle']
    ring   = fs['ring']
    pinky  = fs['pinky']
    ext    = _count_extended(fs)
    orient = _hand_orientation(lms)
    palm   = _palm_size(lms)
    
    debug = {
        'ext': ext,
        'ext_names': ", ".join([f.capitalize() for f, s in fs.items() if s]),
        'orient': orient
    }

    def _result(letter, conf):
        return letter, conf, debug

    # ─── PHASE 1: Shape-based detection (independent of extension count) ───
    # These letters have distinctive shapes that don't fit neatly into
    # extension-count categories because fingers are partially curled.

    is_f, f_conf = _detect_F_shape(lms, fs)
    if is_f and f_conf > 0.78:
        return _result('F', f_conf)

    # O: Thumb + index tips touching, all fingers curled inward
    is_o, o_conf = _detect_O_shape(lms)
    if is_o and o_conf > 0.76 and ext <= 2:
        return _result('O', o_conf)

    # C: Curved hand with gap between thumb and fingers
    is_c, c_conf = _detect_C_shape(lms)
    if is_c and c_conf > 0.73 and ext <= 4:
        # Distinguish from other letters: C should NOT have ALL fully straight fingers
        straight_count = sum(
            1 for f in ['index', 'middle', 'ring', 'pinky']
            if _finger_curl_angle(lms, f) > 170
        )
        if straight_count <= 1:
            return _result('C', c_conf)

    # X: index finger hooked (bent at DIP but partially extended from MCP)
    if not middle and not ring and not pinky:
        dip_angle = _angle_at(lms[6], lms[7], lms[8])
        pip_angle = _finger_curl_angle(lms, 'index')
        # Index is partially out from fist but DIP is bent (hooked)
        if dip_angle < 130 and pip_angle > 100:
            if lms[8].y > lms[7].y or dip_angle < 110:
                return _result('X', 0.82)

    # ─── PHASE 2: Extension-count based classification ───

    # ────────── 0 or 1 extended: A, E, M, N, S, T ──────────

    if ext <= 1:
        thumb_tip = lms[4]
        index_tip = lms[8]
        index_pip = lms[6]
        middle_tip = lms[12]
        middle_pip = lms[10]
        ring_tip = lms[16]
        ring_pip = lms[14]
        pinky_tip = lms[20]

        # M: Fist, thumb tucked under 3 fingers
        # Thumb tip is below index, middle, AND ring fingertips
        # Thumb tip is near ring PIP (exits between ring and pinky)
        thumb_under_3 = (thumb_tip.y > index_tip.y - 0.05 and
                         thumb_tip.y > middle_tip.y - 0.05 and
                         thumb_tip.y > ring_tip.y - 0.05)
        
        # M fallback: thumb is close to the palm area under middle/ring fingers
        thumb_tucked = thumb_under_3 or (_dist2d(thumb_tip, lms[9]) < 0.12 or 
                                        _dist2d(thumb_tip, lms[13]) < 0.12)
        
        if thumb_tucked:
            d_to_ring = _dist2d(thumb_tip, ring_pip)
            d_to_pinky = _dist2d(thumb_tip, lms[17])
            d_to_pinky_pip = _dist2d(thumb_tip, lms[18])
            # Also check distance to ring MCP as alternative anchor
            d_to_ring_mcp = _dist2d(thumb_tip, lms[13])
            if d_to_ring < 0.13 or d_to_pinky < 0.13 or d_to_pinky_pip < 0.11 or d_to_ring_mcp < 0.11:
                return _result('M', 0.82)

        # M alternative: thumb between ring and pinky, 3 fingers over thumb
        # Check if index, middle, ring tips are all roughly at same height (curled over)
        tips_y = [index_tip.y, middle_tip.y, ring_tip.y]
        tips_y_range = max(tips_y) - min(tips_y)
        if tips_y_range < 0.06:
            # All three fingertips at similar height, thumb somewhere underneath
            if thumb_tip.y > min(tips_y) - 0.04:
                d_ring_pinky_mid_x = (lms[16].x + lms[20].x) / 2
                d_ring_pinky_mid_y = (lms[16].y + lms[20].y) / 2
                # Thumb should be near the ring-pinky area
                thumb_near = (abs(thumb_tip.x - d_ring_pinky_mid_x) < 0.10 and
                              abs(thumb_tip.y - d_ring_pinky_mid_y) < 0.10)
                if thumb_near:
                    return _result('M', 0.78)

        # N: Fist, thumb tucked under 2 fingers
        # Thumb tip is below index and middle tips, but not ring
        thumb_under_2 = (thumb_tip.y > index_tip.y - 0.01 and
                         thumb_tip.y > middle_tip.y - 0.01 and
                         thumb_tip.y <= ring_tip.y + 0.02)
        if thumb_under_2:
            d_to_mid = _dist2d(thumb_tip, middle_pip)
            d_to_ring_mcp = _dist2d(thumb_tip, lms[13])
            if d_to_mid < 0.09 or d_to_ring_mcp < 0.09:
                return _result('N', 0.80)

        # T: Thumb tucked between index and middle
        if (_tips_close(lms, 4, 6, 0.07) or _tips_close(lms, 4, 7, 0.07)):
            return _result('T', 0.82)

        # S: Fist with thumb across front of curled fingers
        # Thumb crosses over the fingers (thumb tip in front, lower z)
        if thumb_tip.y < index_pip.y + 0.02:
            # Thumb is in front of (or above) the curled fingers
            idx_x = lms[5].x
            pnk_x = lms[17].x
            min_x = min(idx_x, pnk_x) - 0.04
            max_x = max(idx_x, pnk_x) + 0.04
            if min_x <= thumb_tip.x <= max_x:
                return _result('S', 0.80)

        # E: All fingertips curled tightly down toward palm
        tips_below_pips = sum(1 for t, p in zip(
            [index_tip, middle_tip, ring_tip, pinky_tip],
            [index_pip, middle_pip, ring_pip, lms[18]]
        ) if t.y > p.y - 0.01)
        if tips_below_pips >= 3:
            return _result('E', 0.78)

        # A: Default fist with thumb on the side
        return _result('A', 0.75)

    # ────────── 1 extended ──────────

    if ext == 1:
        # A: thumb only extended (thumb sticking out from fist)
        if thumb and not index and not middle and not ring and not pinky:
            return _result('A', 0.88)

        # D: index only up, other fingers curled
        if index and not thumb and not middle and not ring and not pinky:
            return _result('D', 0.90)

        # I: pinky only up
        if pinky and not thumb and not index and not middle and not ring:
            return _result('I', 0.92)

        # J: Like I (pinky extended) but with hand angled sideways
        if pinky and not index and not middle and not ring:
            if orient == 'side' or orient == 'down':
                return _result('J', 0.72)

        # X revisited: index partially out but hooked
        if not middle and not ring and not pinky:
            dip_angle = _angle_at(lms[6], lms[7], lms[8])
            if dip_angle < 140:
                return _result('X', 0.80)

    # ────────── 2 extended ──────────

    if ext == 2:
        # Y: thumb + pinky extended (unique combo)
        if thumb and pinky and not index and not middle and not ring:
            return _result('Y', 0.95)

        # Handle thumb + index combinations (L, G, Q)
        if thumb and index and not middle and not ring and not pinky:
            if orient == 'up':
                # L: thumb + index forming L shape
                return _result('L', 0.90)
            elif _index_pointing_sideways(lms):
                # G: index pointing sideways
                return _result('G', 0.85)
            elif orient == 'down':
                # Q: pointing downward
                return _result('Q', 0.82)
            else:
                # Default L for thumb+index
                return _result('L', 0.78)

        # Handle index + middle combinations (H, P, R, K, U, V)
        if index and middle and not ring and not pinky:
            # H: index + middle pointing sideways
            if _index_pointing_sideways(lms):
                return _result('H', 0.85)

            # P: like K but hand pointing down
            if orient == 'down':
                return _result('P', 0.82)

            # R: index + middle crossed (tips overlapping)
            tips_dist = _dist2d(lms[8], lms[12])
            if tips_dist < 0.025:
                return _result('R', 0.82)

            # K: index + middle with thumb touching between them
            if thumb or _tips_close(lms, 4, 10, 0.09):
                if _tips_close(lms, 4, 10, 0.09) or _tips_close(lms, 4, 9, 0.09):
                    return _result('K', 0.80)

            # U: index + middle together (tips close)
            if not thumb:
                if tips_dist < 0.06:
                    return _result('U', 0.88)
                # V: index + middle spread (peace sign)
                else:
                    return _result('V', 0.92)

            # Fallback for 2 fingers (index+middle): U if close, V if spread
            if tips_dist < 0.05:
                return _result('U', 0.72)
            return _result('V', 0.72)

        # D variant when extension detection is slightly off
        if index and not middle and not ring and not pinky:
            return _result('D', 0.78)

    # ────────── 3 extended ──────────

    if ext == 3:
        # W: index + middle + ring up, thumb and pinky down
        if index and middle and ring and not pinky and not thumb:
            return _result('W', 0.90)

        # F detected by shape earlier, but catch fallback
        if middle and ring and pinky:
            if _tips_close(lms, 4, 8, 0.10):
                return _result('F', 0.82)

        # P: thumb + index + middle, hand pointing down
        if thumb and index and middle and not ring and not pinky:
            if orient == 'down':
                return _result('P', 0.82)

        # K variant: thumb + index + middle with thumb touching
        if index and middle and thumb and not ring and not pinky:
            if _tips_close(lms, 4, 10, 0.09):
                return _result('K', 0.78)

        # Z: static approximation (index + thumb + another, hand sideways)
        if thumb and index and not ring and not pinky:
            if orient == 'side':
                return _result('Z', 0.65)

        # Default 3 fingers: W
        if index and middle and ring:
            return _result('W', 0.70)
        if thumb and index and middle:
            return _result('W', 0.65)

    # ────────── 4 extended ──────────

    if ext == 4:
        # B: four fingers up, thumb folded
        if index and middle and ring and pinky and not thumb:
            return _result('B', 0.90)

        # F variant: thumb+index circle, 3 others up
        if _tips_close(lms, 4, 8, 0.09) and middle and ring and pinky:
            return _result('F', 0.85)

        # B variant
        if index and middle and ring and pinky:
            return _result('B', 0.80)

    # ────────── 5 extended (open hand) ──────────

    if ext == 5:
        spread = _fingers_spread(lms)
        if spread:
            return _result('B', 0.82)  # Open hand with spread = 5/B
        else:
            return _result('B', 0.88)  # Fingers together = B

    # ─── Fallback compound checks ───

    # O fallback
    if _tips_close(lms, 4, 8, 0.07):
        mid_curl = _finger_curl_angle(lms, 'middle')
        if mid_curl < 155:
            return _result('O', 0.72)

    # C fallback (wider thresholds)
    is_c, c_conf = _detect_C_shape(lms)
    if is_c:
        return _result('C', max(c_conf - 0.05, 0.70))

    # D fallback
    if index and not middle and not ring and not pinky:
        return _result('D', 0.68)

    return _result(None, 0.0)
