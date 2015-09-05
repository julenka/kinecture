#!/usr/bin/env python
# coding=utf-8
""" Trying to compute own speakerXY features to remove extraneous zeros, maybe that will improve accuracy
"""
__author__ = 'julenka'

import math

class ClassroomParams:
    def __init__(self, classroom, roomX, roomY, Ltheta, Rtheta, dxL, dyL, dxR, dyR):
        self.classroom = classroom
        self.roomX = roomX
        self.roomY = roomY
        self.Ltheta = Ltheta
        self.Rtheta = Rtheta
        self.dxL = dxL
        self.dyL = dyL
        self.dxR = dxR
        self.dyR = dyR

        self.Rx = roomX - dxR
        self.Ry = roomY - dyR

        self.Lx = dxL
        self.Ly = roomY - dyL

# classroom_param_map = { 9: ClassroomParams(9, 226, 356, 317, 220, 7, 10, 22, 12),
#                        10: ClassroomParams(10, 456, 317, 313, 221, 12, 6, 13, 8),
#                        11: ClassroomParams(11, 390, 334, 312, 222, 9, 11, 6, 48)}
# I had to modify the classroom params to make the y position of the kinects (dyR, dyL) match up, otherwise
# speakerX and speakerY positions could be too large because the lines wouldn't intersect.
classroom_param_map = { 9: ClassroomParams(9, 226, 356, 317, 220,  7, 10, 22, 10),
                       10: ClassroomParams(10, 456, 317, 313, 221, 12, 6, 13, 6),
                       11: ClassroomParams(11, 390, 334, 312, 222, 9, 11, 6, 11)}



def gen_my_speaker_features(data):
    result = data.copy()
    for row_num, row in result.iterrows():
        my_speaker_x, my_speaker_y = get_speakerxy_from_angles(row.session,
                                                               row['angleLeft'],
                                                               row['angleRight'])
        result.loc[row_num, 'mySpeakerX'] = my_speaker_x
        result.loc[row_num, 'mySpeakerY'] = my_speaker_y

    return result

def get_speakerxy_from_angles(classroom_number, left_angle, right_angle):
    classroom_params = classroom_param_map[classroom_number]

    left_angle_2 = classroom_params.Ltheta + left_angle
    right_angle_2 = classroom_params.Rtheta + right_angle

    # Clamp these bad boys
    left_angle_2 = max(271, min(359, left_angle_2))
    right_angle_2 = max(181, min(269, right_angle_2))

    # y = mx + b
    left_m = math.tan(math.radians(left_angle_2))
    left_b = classroom_params.Ly - left_m * classroom_params.Lx

    right_m = math.tan(math.radians(right_angle_2))
    right_b = classroom_params.Ry - right_m * classroom_params.Rx

    a = left_m
    b = left_b
    c = right_m
    d = right_b

    x = (b - d) / (c - a)
    y = a * (b - d) / (c - a) + b

    x = max(0, min(classroom_params.roomX, x)) / classroom_params.roomX
    y = max(0, min(classroom_params.roomY, y)) / classroom_params.roomY

    return x, y
