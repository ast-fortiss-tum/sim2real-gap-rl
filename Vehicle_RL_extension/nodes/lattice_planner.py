#!/usr/bin/env python3

import os
import pprint
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
import time

import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from mixed_reality.msg import Waypoint, WaypointList, Obstacles, SimPose
import tf

from mixed_reality.lattice_planner import cubic_spline, quintic_polynomial, quartic_polynomial, env, draw, frenet
from mixed_reality.for_conversions.for_conversions import For_convertion_utils

#TODO: revert parameters to the ones on the reference/original file
class C:
    # Parameter
    K_SIZE=7.33
    # Parameter
    MAX_SPEED = 3.5*7.33 / 3.6
    MAX_ACCEL = 2.0
    MAX_CURVATURE = 2.5

    ROAD_WIDTH = 3.
    ROAD_SAMPLE_STEP = 1.0  #precision of reference path

    T_STEP = 0.5   #smaller means more points on path
    MAX_T = 10.0 #max and min distance for path endpoints
    MIN_T = 6.0

    TARGET_SPEED = 2.0*7.33 / 3.6
    SPEED_SAMPLE_STEP = 5.0 / 3.6

    # cost weights for Cruising
    K_JERK = 0.1
    K_TIME = 0.1
    K_V_DIFF = 1.0
    K_OFFSET = 100
    K_COLLISION = 200

    # cost weights for Stopping
    # K_JERK = 0.1
    # K_TIME = 0.1
    # K_V_DIFF = 200
    # K_OFFSET = 1.0
    # K_COLLISION = 500

    # parameters for vehicle
    
    RF = 0.22 * K_SIZE  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.05 * K_SIZE  # [m] distance from rear to vehicle back end of vehicle
    W = 0.16 * K_SIZE  # [m] width of vehicle
    WD = 0.9 * W  # [m] distance between left-right wheels
    WB = 0.2 * K_SIZE  # [m] Wheel base
    TR = 0.05 * K_SIZE  # [m] Tyre radius
    TW = 0.03 * K_SIZE  # [m] Tyre width
    MAX_STEER = 0.8  # [rad] maximum steering angle


class Obstacle:
    def __init__(self, x, y,yaw, width, height):
        self.x = x
        self.y = y
        self.yaw=yaw
        self.width = width
        self.height = height


class Path:
    def __init__(self):
        self.t = []

        self.l = []
        self.l_v = []
        self.l_a = []
        self.l_jerk = []

        self.s = []
        self.s_v = []
        self.s_a = []
        self.s_jerk = []

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.curv = []

        self.cost = 0.0


def sampling_paths_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    PATHS = dict()

    for s1_v in np.arange(C.TARGET_SPEED * 0.6, C.TARGET_SPEED * 1.4, C.TARGET_SPEED * 0.05):

        for t1 in np.arange(4.5, 5.5, 0.2):
            path_pre = Path()
            path_lon = quartic_polynomial.QuarticPolynomial(s0, s0_v, s0_a, s1_v, 0.0, t1)

            path_pre.t = list(np.arange(0.0, t1, C.T_STEP))
            path_pre.s = [path_lon.calc_xt(t) for t in path_pre.t]
            path_pre.s_v = [path_lon.calc_dxt(t) for t in path_pre.t]
            path_pre.s_a = [path_lon.calc_ddxt(t) for t in path_pre.t]
            path_pre.s_jerk = [path_lon.calc_dddxt(t) for t in path_pre.t]

            for l1 in np.arange(-C.ROAD_WIDTH, C.ROAD_WIDTH, C.ROAD_SAMPLE_STEP):
                path = copy.deepcopy(path_pre)
                path_lat = quintic_polynomial.QuinticPolynomial(l0, l0_v, l0_a, l1, 0.0, 0.0, t1)

                path.l = [path_lat.calc_xt(t) for t in path_pre.t]
                path.l_v = [path_lat.calc_dxt(t) for t in path_pre.t]
                path.l_a = [path_lat.calc_ddxt(t) for t in path_pre.t]
                path.l_jerk = [path_lat.calc_dddxt(t) for t in path_pre.t]

                path.x, path.y = SL_2_XY(path.s, path.l, ref_path)
                path.yaw, path.curv, path.ds = calc_yaw_curv(path.x, path.y)

                l_jerk_sum = sum(np.abs(path.l_jerk))
                s_jerk_sum = sum(np.abs(path.s_jerk))
                v_diff = abs(C.TARGET_SPEED - path.s_v[-1])
                path.cost = C.K_JERK * (l_jerk_sum + s_jerk_sum) + \
                            C.K_V_DIFF * v_diff + \
                            C.K_TIME * t1 * 2 + \
                            C.K_OFFSET * abs(path.l[-1]) + \
                            C.K_COLLISION * is_path_collision(path)

                PATHS[path] = path.cost
    return PATHS


def sampling_paths_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    PATHS = dict()

    for s1_v in [-1.0, 0.0, 1.0, 2.0]:

        for t1 in np.arange(1.0, 16.0, 1.0):
            path_pre = Path()
            path_lon = quintic_polynomial.QuinticPolynomial(s0, s0_v, s0_a, 55.0, s1_v, 0.0, t1)

            path_pre.t = list(np.arange(0.0, t1, C.T_STEP))
            path_pre.s = [path_lon.calc_xt(t) for t in path_pre.t]
            path_pre.s_v = [path_lon.calc_dxt(t) for t in path_pre.t]
            path_pre.s_a = [path_lon.calc_ddxt(t) for t in path_pre.t]
            path_pre.s_jerk = [path_lon.calc_dddxt(t) for t in path_pre.t]

            for l1 in [0.0]:
                path = copy.deepcopy(path_pre)
                path_lat = quintic_polynomial.QuinticPolynomial(l0, l0_v, l0_a, l1, 0.0, 0.0, t1)

                path.l = [path_lat.calc_xt(t) for t in path_pre.t]
                path.l_v = [path_lat.calc_dxt(t) for t in path_pre.t]
                path.l_a = [path_lat.calc_ddxt(t) for t in path_pre.t]
                path.l_jerk = [path_lat.calc_dddxt(t) for t in path_pre.t]

                path.x, path.y = SL_2_XY(path.s, path.l, ref_path)
                path.yaw, path.curv, path.ds = calc_yaw_curv(path.x, path.y)

                if path.yaw is None:
                    continue

                l_jerk_sum = sum(np.abs(path.l_jerk))
                s_jerk_sum = sum(np.abs(path.s_jerk))
                v_diff = (path.s_v[-1]) ** 2

                path.cost = C.K_JERK * (l_jerk_sum + s_jerk_sum) + \
                            C.K_V_DIFF * v_diff + \
                            C.K_TIME * t1 * 2 + \
                            C.K_OFFSET * abs(path.l[-1]) + \
                            50.0 * sum(np.abs(path.s_v))

                PATHS[path] = path.cost
                print(path.cost)
    return PATHS


def SL_2_XY(s_set, l_set, ref_path):
    pathx, pathy = [], []

    for i in range(len(s_set)):
        x_ref, y_ref = ref_path.calc_position(s_set[i])

        if x_ref is None:
            break

        yaw = ref_path.calc_yaw(s_set[i])
        x = x_ref + l_set[i] * math.cos(yaw + math.pi / 2.0)
        y = y_ref + l_set[i] * math.sin(yaw + math.pi / 2.0)

        pathx.append(x)
        pathy.append(y)

    return pathx, pathy


def calc_yaw_curv(x, y):
    yaw, curv, ds = [], [], []

    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        ds.append(math.hypot(dx, dy))
        yaw.append(math.atan2(dy, dx))

    if len(yaw) == 0:
        return None, None, None

    yaw.append(yaw[-1])
    ds.append(ds[-1])

    for i in range(len(yaw) - 1):
        curv.append((yaw[i + 1] - yaw[i]) / ds[i])

    return yaw, curv, ds


def is_path_collision(path):
    # global obstacle_array
    # index = range(0, len(path.x), 5)
    # x = [path.x[i] for i in index]
    # y = [path.y[i] for i in index]

    # yaw = [path.yaw[i] for i in index]

    # for ix, iy, iyaw in zip(x, y, yaw):
    #     d = 1.8
    #     dl = (C.RF - C.RB) / 2.0
    #     r = math.hypot((C.RF + C.RB) / 2.0, C.W / 2.0) + d

    #     cx = ix + dl * math.cos(iyaw)
    #     cy = iy + dl * math.sin(iyaw)

    #     for i in range(len(obstacle_array)):
    #         xo = obstacle_array[i].x - cx
    #         yo = obstacle_array[i].y - cy
    #         dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
    #         dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

    #         if abs(dx) < r + obstacle_array[i].width / 2 and abs(dy) < C.W / 2 + d + obstacle_array[i].height / 2:
    #             print("COLLISION")
    #             return 1.0
    # print("no collision")
    # return 0.0
    
    index = range(0, len(path.x), 5)
    x = [path.x[i] for i in index]
    y = [path.y[i] for i in index]
    yaw = [path.yaw[i] for i in index]

    for ix, iy, iyaw in zip(x, y, yaw):
        d = 1.8
        dl = (C.RF - C.RB) / 2.0
        r = math.hypot((C.RF + C.RB) / 2.0, C.W / 2.0) + d

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        for i in range(len(C.obs)):
            xo = C.obs[i][0] - cx
            yo = C.obs[i][1] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return 1.0

    return 0.0


def verify_path(path):

    if any([v > C.MAX_SPEED for v in path.s_v]) or \
            any([abs(a) > C.MAX_ACCEL for a in path.s_a]) or \
            any([abs(curv) > C.MAX_CURVATURE for curv in path.curv]):
        return False
    #TODO: collision check added by myself
    if is_path_collision(path) == 1.0:
        return False

    return True


def extract_optimal_path(paths):
    if len(paths) == 0:
        return None
        
    while len(paths) > 1:
        path = min(paths, key=paths.get)
        paths.pop(path)
        # if verify_path(path) is False:
        if verify_path(path) is False:
            print("REJECTED")
            continue
        else:
            print("accepted")
            return path
    return list(paths.items())[-1][0]


def lattice_planner_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    paths = sampling_paths_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)
    path = extract_optimal_path(paths)
    return path


def lattice_planner_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    paths = sampling_paths_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)
    path = extract_optimal_path(paths)

    return path


def get_reference_line(x, y):
    index = range(0, len(x), 3)
    x = [x[i] for i in index]
    y = [y[i] for i in index]

    cubicspline = cubic_spline.Spline2D(x, y)
    s = np.arange(0, cubicspline.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = cubicspline.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(cubicspline.calc_yaw(i_s))
        rk.append(cubicspline.calc_curvature(i_s))

    return rx, ry, ryaw, rk, cubicspline


def pi_2_pi(theta):
    if theta > math.pi:
        return theta - 2.0 * math.pi

    if theta < -math.pi:
        return theta + 2.0 * math.pi

    return theta


obstacle_dict = {}  #obstacle_dict keeps "name":[x,y]
obstacle_array = np.array([])   #obstacle_array keeps Obstacle objects
#NOTE: the pose and speed will be in the frame of reference of the simulator, angle is in degrees
pose = [100,100,100]
speed = 0.0
SIZE_FACTOR=7.33
X_MAP_SHIFT=48
Y_MAP_SHIFT=50
ANGLE_SHIFT=0
for_conversions = For_convertion_utils(SIZE_FACTOR,X_MAP_SHIFT,Y_MAP_SHIFT,ANGLE_SHIFT)
TRACKING = False
got_pose = False

def new_pose(msg):
    global pose
    global for_conversions
    global got_pose

    if TRACKING:
        position = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        )
        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        r = math.degrees(euler[2])
        pose = for_conversions.real2sim_xyp([position[0], position[1], r])
        pose[2] = 90-pose[2]
    else:
        #TODO: sumar 180 es provisional, checar si dejarlo o no
        pose = [msg.x,msg.z,-msg.yaw+90]    #TODO: check where the sim is sending its position values (in what order, etc)
    got_pose = True

def new_speed(msg):
    global speed
    if TRACKING:
        speed = msg.data*SIZE_FACTOR
    else:
        speed = msg.data
        # print(f"real speed: {speed/SIZE_FACTOR}")

def new_obstacles(msg):
    global obstacle_dict
    global obstacle_array
    for obstacle in msg.data:
        obstacle_dict[obstacle.name] = [obstacle.x, obstacle.y]
    new_array = np.array([])
    for o in obstacle_dict:
        obstacle = Obstacle(obstacle_dict[o][0], obstacle_dict[o][1], (3.14/2), C.W,C.RF)
        new_array = np.append(new_array,obstacle)
    obstacle_array = new_array


def main_Crusing():
    global obstacle_dict
    global obstacle_array
    global for_conversions
    global speed
    global pose
    global got_pose
    ENV = env.ENVCrusing()
    if TRACKING:
        C.K_SIZE = 1
    K_SIZE = 1

    #TODO: change this manual read to a request to the environment server
    f = open("map.json", "r")
    map_data = json.loads(f.read())
    f.close()
    waypoints = map_data["right_lane"]
    center_line = map_data["sim_waypoint_list"]
    left_lane = map_data["left_margin"]
    right_lane = map_data["right_margin"]

    #waypoints=[[55.33, 38.272], [54.52094804583332, 38.3049425405176], [53.78133299980349, 38.32858365783024], [53.106636642651424, 38.343842011019866], [52.49234075511813, 38.35163625916847], [51.933927117944556, 38.35288506135804], [51.42687751187168, 38.348507076670515], [50.96667371764047, 38.33942096418788], [50.548797515991886, 38.32654538299211], [50.1687306876669, 38.310798992165175], [49.82195501340649, 38.29310045078905], [49.5039522739516, 38.27436841794571], [49.21020425004322, 38.25552155271712], [48.93619272242231, 38.237478514185256], [48.67739947182984, 38.22115796143209], [48.42930627900678, 38.207478553539595], [48.18739492469409, 38.19735894958975], [47.94714718963274, 38.19171780866451], [47.704044854563705, 38.19147378984586], [47.45356970022795, 38.197545552215765], [47.19120350736644, 38.210851754856215], [46.912428056720145, 38.232311056849156], [46.61272512903004, 38.26284211727659], [46.28824739149286, 38.30342521311812], [45.94098670823532, 38.35557692527742], [45.57594150861188, 38.42108997412543], [45.19813506962354, 38.501759362177445], [44.81259066827124, 38.59938009194875], [44.424331581556, 38.71574716595467], [44.03838108647876, 38.85265558671047], [43.65976246004052, 39.01190035673145], [43.29349897924226, 39.19527647853291], [42.944613921084944, 39.40457895463015], [42.61813056256956, 39.64160278753846], [42.318818905330296, 39.90800034575079], [42.048010484621784, 40.20348759827264], [41.80455878950996, 40.52638498283485], [41.58726260158157, 40.87498212821943], [41.39492070242332, 41.247568663208355], [41.22633187362194, 41.64243421658366], [41.080294896764144, 42.05786841712734], [40.95560855343667, 42.492160893621396], [40.85107162522622, 42.94360127484783], [40.765482893719536, 43.410479189588656], [40.69764114050332, 43.89108426662587], [40.64633235009316, 44.38371381298417], [40.610048174367805, 44.88684173527033], [40.586985932569284, 45.39911853967319], [40.57533014686844, 45.91920241062427], [40.57326533943609, 46.44575153255511], [40.578976032443116, 46.97742408989724], [40.590646748060315, 47.51287826708219], [40.60646200845857, 48.0507722485415], [40.62460633580869, 48.58976421870669], [40.64326425228154, 49.128512362009296], [40.66062028004796, 49.665674862880856], [40.67487909666587, 50.199927757667176], [40.68515834407849, 50.73075570831247], [40.69184246763206, 51.258765402632186], [40.69540922465003, 51.78464617619302], [40.696336372455875, 52.3090873645617], [40.69510166837305, 52.83277830330492], [40.69218286972502, 53.35640832798939], [40.68805773383525, 53.88066677418185], [40.683204018027205, 54.40624297744898], [40.67809947962433, 54.93382627335751], [40.6732218759501, 55.46410599747413], [40.66904747133635, 55.9977660039534], [40.665871878126964, 56.53482689607991], [40.66363985563141, 57.07402114528332], [40.662255852384995, 57.61393322486521], [40.661624316923046, 58.153147608127234], [40.66164969778086, 58.69024876837101], [40.66223644349374, 59.22382117889818], [40.66328900259699, 59.752449313010345], [40.66471182362593, 60.27471764400915], [40.66640935511585, 60.78921064519623], [40.66828604560208, 61.29451278987318], [40.67024634361991, 61.78920855134164], [40.67219469770466, 62.27188240290326], [40.67403555639162, 62.74111881785964], [40.67567336821612, 63.19550226951242], [40.677012581713456, 63.633617231163214], [40.67795764541893, 64.05404817611364], [40.67841300786785, 64.45537957766537], [40.678283117595534, 64.83619590912], [40.67747242313729, 65.19508164377913], [40.67588537302841, 65.53062125494444], [40.67342641580421, 65.84139921591752], [40.67, 66.12599999999999]]
    wx = [point[0]/K_SIZE for point in waypoints]
    wy = [point[1]/K_SIZE for point in waypoints]

    cx = [point[0]/K_SIZE for point in center_line]
    cy = [point[1]/K_SIZE for point in center_line]

    #left_lane = [[55.20794875342281, 35.27448377932513], [54.41200077143069, 35.30695012048921], [53.6994973087895, 35.32973278451455], [53.05369198332356, 35.34434617374535], [52.46995563559294, 35.35176073641368], [51.94352348534089, 35.352944721617924], [51.469438115878916, 35.348855260008904], [51.04247945768565, 35.34042476383173], [50.65708612827037, 35.328542623703854], [50.307279937694375, 35.3140345137871], [49.98661664339574, 35.2976458764601], [49.68819974536164, 35.28004191430688], [49.40480489639883, 35.26184083609993], [49.12915942550908, 35.24369385917416], [48.854389847740535, 35.22640687872313], [48.57458108875353, 35.21106424090563], [48.285298967621294, 35.199082988630906], [47.983863913457455, 35.19213189441894], [47.66919972248722, 35.19191507008499], [47.341242340112316, 35.19991145908137], [47.00010199166572, 35.21720157978887], [46.64528327084289, 35.24445872656409], [46.2745475937944, 35.28216173174648], [45.87931746481253, 35.331656328205064], [45.45324927161776, 35.395799297901505], [44.99775611322375, 35.477744386118395], [44.51673049611575, 35.58071462633185], [44.01376015987991, 35.70841650498975], [43.49220652653013, 35.86520526907044], [42.95535520393836, 36.05628607951973], [42.406679252140755, 36.28793804365331], [41.85029651678887, 36.567708021260174], [41.29170841989968, 36.90444909239871], [40.73962895448888, 37.30728213810211], [40.21572305213209, 37.77404841556637], [39.74445738388303, 38.28697971301097], [39.33389566558475, 38.829869763591645], [38.98144517365749, 39.393424459311106], [38.682521943220564, 39.970502054575995], [38.43169520892545, 40.55599046603325], [38.22342697741633, 41.14648482447686], [38.05252020717205, 41.739838877752334], [37.914326474681424, 42.3347358533025], [37.804794661660715, 42.93034432457531], [37.72043612551989, 43.526035435958946], [37.65828813675138, 44.12042943788908], [37.615451425478646, 44.711485825454254], [37.58888022548739, 45.298049615959194], [37.57571823961618, 45.87971176495256], [37.57336339157698, 46.455980049564076], [37.579418721360284, 47.02622076988066], [37.591650864164635, 47.589648613976586], [37.6079591539506, 48.14532306347987], [37.62635425717835, 48.69214765126249], [37.644945341894676, 49.22886979134787], [37.66193659970435, 49.75413508783306], [37.67569420948184, 50.26898879204658], [37.6855596795437, 50.778784214723295], [37.6919971426274, 51.28792595240343], [37.69544606865983, 51.7974714207315], [37.69634288552445, 52.30820264243259], [37.695129140323296, 52.82088065570283], [37.692252605644555, 53.33624480629904], [37.688168127178734, 53.855012400596586], [37.68333818233723, 54.37787870813712], [37.678233134329474, 54.905517296089904], [37.67333121489716, 55.43857635711526], [37.6691195807439, 55.977163825405434], [37.66591095107731, 56.519748410337705], [37.663657635570566, 57.063966803996976], [37.662261809378535, 57.608331338126], [37.66162534740354, 58.15146167115417], [37.66165060638612, 58.69196913317576], [37.66224032381785, 59.22845732652209], [37.66329754235989, 59.75952242911462], [37.66472555458655, 60.28375320618094], [37.666427864928686, 60.79973071357419], [37.6683081675333, 61.30602765011388], [37.67027034069487, 61.8012072833333], [37.67221846093807, 62.28382182735665], [37.67405684332149, 62.75241008485172], [37.675690120077576, 63.205494063107345], [37.677023379105115, 63.641574113242044], [37.67796240054794, 64.0591218786405], [37.6784140606586, 64.45656989519092], [37.67828703187694, 64.83229591089969], [37.67749302845373, 65.18459858413947], [37.67594910205193, 65.51165854361095], [37.67364381202733, 65.80528370570941]]
    bx1 = [point[0]/K_SIZE for point in left_lane]
    by1 = [point[1]/K_SIZE for point in left_lane]
    
    #right_lane = [[55.452051246577184, 41.269516220674866], [54.62989532023595, 41.30293496054599], [53.863168690817474, 41.327434531145926], [53.15958130197929, 41.34333784829438], [52.51472587464332, 41.351511781923264], [51.92433075054822, 41.35282540109815], [51.384316907864445, 41.348158893332126], [50.890867977595285, 41.33841716454403], [50.4405089037134, 41.32454814228036], [50.03018143763943, 41.30756347054325], [49.65729338341723, 41.288555025117994], [49.31970480254155, 41.26869492158454], [49.01560360368761, 41.24920226933431], [48.743226019335545, 41.23126316919635], [48.50040909591914, 41.21590904414105], [48.28403146926003, 41.20389286617356], [48.089490881766885, 41.19563491054859], [47.91043046580802, 41.19130372291007], [47.73888998664019, 41.19103250960673], [47.56589706034359, 41.19517964535016], [47.382305023067154, 41.20450192992356], [47.1795728425974, 41.22016338713422], [46.950902664265676, 41.243522502806705], [46.69717731817319, 41.27519409803117], [46.428724144852886, 41.31535455265333], [46.154126904000016, 41.36443556213246], [45.87953964313133, 41.42280409802304], [45.611421176662574, 41.490343678907756], [45.35645663658186, 41.5662890628389], [45.121406969019155, 41.64902509390121], [44.91284566794028, 41.73586266980959], [44.736701441695644, 41.82284493580565], [44.59751942227021, 41.9047088168616], [44.496632170650244, 41.97592343697481], [44.4219147585285, 42.041952275935216], [44.35156358536054, 42.119995483534304], [44.275221913435175, 42.22290020207805], [44.19308002950565, 42.35653979712775], [44.107319461626076, 42.524635271840715], [44.02096853831843, 42.72887796713407], [43.93716281611196, 42.96925200977782], [43.858696899701286, 43.24448290949046], [43.78781677577101, 43.55246669639316], [43.72617112577836, 43.890614054602004], [43.674846155486755, 44.25613309729279], [43.63437656343494, 44.64699818807926], [43.60464492325696, 45.06219764508641], [43.58509163965118, 45.500187463387185], [43.5749420541207, 45.958693056295985], [43.5731672872952, 46.43552301554614], [43.57853334352595, 46.92862740991382], [43.589642631955996, 47.4361079201878], [43.60496486296654, 47.956221433603126], [43.62285841443904, 48.487380786150894], [43.641583162668404, 49.028154932670724], [43.65930396039157, 49.57721463792865], [43.6740639838499, 50.130866723287774], [43.684757008613275, 50.68272720190165], [43.69168779263671, 51.229604852860945], [43.69537238064023, 51.771820931654545], [43.6963298593873, 52.309972086690806], [43.6950741964228, 52.84467595090701], [43.69211313380549, 53.37657184967974], [43.68794734049177, 53.90632114776711], [43.68306985371718, 54.43460724676085], [43.677965824919184, 54.96213525062512], [43.673112537003036, 55.489635637833], [43.668975361928794, 56.01836818250137], [43.66583280517662, 56.54990538182212], [43.66362207569225, 57.08407548656966], [43.662249895391454, 57.619535111604414], [43.66162328644255, 58.154833545100296], [43.6616487891756, 58.688528403566266], [43.66223256316963, 59.219185031274264], [43.66328046283409, 59.74537619690607], [43.664698092665304, 60.26568208183736], [43.66639084530302, 60.77869057681826], [43.668263923670864, 61.28299792963248], [43.670222346544946, 61.77720981934998], [43.672170934471254, 62.25994297844987], [43.67401426946176, 62.72982755086757], [43.67565661635466, 63.18551047591749], [43.677001784321796, 63.625660349084384], [43.677952890289916, 64.04897447358678], [43.6784119550771, 64.45418926013981], [43.678279203314126, 64.8400959073403], [43.677451817820845, 65.2055647034188], [43.675821644004884, 65.54958396627792], [43.67320901958109, 65.87751472612563]]
    bx2 = [point[0]/K_SIZE for point in right_lane]
    by2 = [point[1]/K_SIZE for point in right_lane]

    #Read obstacles from JSON file
    C.obs = np.array([Obstacle(39.174046199856555, 62.74676445135568,(3.14/2), C.W, C.RF),Obstacle(42.07875468798453, 46.95302574990553, (3.14/2), C.W, C.RF)])
    # for obstacle in map_data["obstacles"]:
    #     obstacle_dict[obstacle[0]] = [obstacle[1],obstacle[2]]
    # obstacle_array = np.array(list(map(lambda o: Obstacle(obstacle_dict[o][0], obstacle_dict[o][1], (3.14/2),C.W,C.RF), obstacle_dict.items())))

    obs_x = [obs.x for obs in obstacle_array]
    obs_y = [obs.y for obs in obstacle_array]

    rx, ry, ryaw, rk, ref_path = get_reference_line(wx, wy)

    acceleration = 0
    prev_speed = None
    prev_speed_time = None
    curvature = 0
    prev_pose = None
    prev_pose_time = None


    #TODO: calculate these starting values based on the initial (possibly tracked) position of the car
    l0 = 0  # current lateral position [m]
    l0_v = 0.0  # current lateral speed [m/s]
    l0_a = 0.0  # current lateral acceleration [m/s^2]
    s0 = 0.0  # current course position
    s0_v = 2.0*7.33 / 3.6  # current speed [m/s]
    s0_a = 0.0

    rospy.init_node("lattice_planner", anonymous=True)
    print("lattice node initialized")
    pub_waypoints = rospy.Publisher("waypoints", WaypointList, queue_size=10)
    rospy.Subscriber("obstacles", Obstacles, new_obstacles)
    if TRACKING:
        rospy.Subscriber("donkey/pose", PoseStamped, new_pose)
        rospy.Subscriber("donkey/speed", Float64, new_speed)
    else:
        rospy.Subscriber("sim/euler", SimPose, new_pose)
        rospy.Subscriber("sim/speed", Float64, new_speed)

    #wait until you receive the initial pose of the car
    while not got_pose:
        pass
    print(f"Received pose is {pose}\nStart of path is {[rx[0],ry[0],ryaw[0]]}")

    while not rospy.is_shutdown():
        #get acceleration of the car
        if prev_speed is None:
            prev_speed = speed
            prev_speed_time = time.time()
        else:
            current_time = time.time()
            time_diff = current_time-prev_speed_time
            if time_diff>0:
                acceleration = (speed-prev_speed)/time_diff
                prev_speed = speed
                prev_speed_time = current_time
        #get curvature of the car trajectory
        if prev_pose is None:
            prev_pose = pose
            prev_pose_time = time.time()
        else:
            current_time = time.time()
            time_diff = current_time-prev_pose_time
            if time_diff>0:
                curvature = (frenet.normalize_angle(math.radians(pose[2]))-frenet.normalize_angle(math.radians(prev_pose[2])))/time_diff
                prev_pose = pose
                prev_pose_time = current_time

        #get values on the reference line
        index = frenet.find_closest_index(rx, ry, pose[0], pose[1])
        rdkappa = 0
        if index<len(rk)-1:
            rdkappa = (rk[index+1]-rk[index])/0.1
        elif index>0:
            rdkappa = (rk[index]-rk[index-1])/0.1
        
        #convert cartesian to frenetcoordinates
        # print(f"index:{index}\nrx:{rx[index]}\nry:{ry[index]}\nryaw:{frenet.normalize_angle(ryaw[index])}\nrk:{rk[index]}\nrdkappa:{rdkappa}\
        #       \nx:{pose[0]}\ny:{pose[1]}\nspeed:{speed}\nacc:{acceleration}\nangle:{math.radians(pose[2])}\ncurv:{curvature}")
        s_condition, d_condition = frenet.cartesian_to_frenet(index*0.1, rx[index], ry[index], frenet.normalize_angle(ryaw[index]), rk[index],rdkappa,
                                                              pose[0], pose[1], speed, acceleration, frenet.normalize_angle(math.radians(pose[2])), curvature)
        l0 = d_condition[0]
        l0_v = d_condition[1]
        l0_a = d_condition[2]
        s0 = s_condition[0]
        s0_v = s_condition[1]
        s0_a = s_condition[2]
        # print(f"d: {d_condition}\ns: {s_condition}")

        path = lattice_planner_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)
        # print("got new path")

        if path is None:
            print("No feasible path found!!")
            pub_waypoints.publish(True, [])
            break

        if np.hypot(path.x[1] - rx[-1], path.y[1] - ry[-1]) <= 0.3*7.33:
            print("Goal")
            pub_waypoints.publish(True, [])
            break

        waypoints = []
        for i in range(0,len(path.x)):
            waypoints.append(Waypoint(path.x[i], path.y[i]))
        pub_waypoints.publish(True, waypoints)
        # print("new waypoints published")

        dy = (path.yaw[2] - path.yaw[1]) / path.ds[1]
        steer = pi_2_pi(math.atan(-C.WB * dy))

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(rx, ry, linestyle=':', color='gray')
        plt.plot(cx, cy, linestyle='--', color='gold')
        plt.plot(bx1, by1, linewidth=1.5, color='k')
        plt.plot(bx2, by2, linewidth=1.5, color='k')
        plt.plot(path.x[1:], path.y[1:], linewidth='2', color='royalblue')
        plt.plot(obs_x, obs_y, 'ok')
        for obs in C.obs:
            draw.draw_car(obs.x, obs.y, obs.yaw, steer, C)
        # for obs in obstacle_array:
        #     draw.draw_car(obs.x, obs.y, obs.yaw, steer, C)
        #draw.draw_car(path.x[1], path.y[1], path.yaw[1], steer, C)
        draw.draw_car(pose[0], pose[1], math.radians(pose[2]), steer, C)
        plt.title("[Crusing Mode]  v :" + str(s0_v * 3.6)[0:4] + " km/h")
        plt.axis("equal")
        plt.pause(0.0001)
        plt.show(block=False)

    plt.pause(0.0001)
    plt.show()


def main_Stopping():
    global obstacle_dict
    global obstacle_array
    ENV = env.ENVCrusing()
    K_SIZE = 7.33

    #TODO: change this manual read to a request to the environment server
    f = open("map.json", "r")
    map_data = json.loads(f.read())
    f.close()
    waypoints = map_data["sim_waypoint_list"]
    left_lane = map_data["left_lane"]
    right_lane = map_data["right_lane"]

    #waypoints=[[55.39102562328859, 39.77075811033743], [54.575421683034634, 39.80393875053179], [53.82225084531048, 39.82800909448808], [53.133108972315355, 39.84358992965713], [52.50353331488073, 39.85157402054587], [51.929128934246386, 39.85285523122809], [51.40559720986806, 39.84833298500132], [50.928770847617876, 39.83891906436595], [50.49465320985264, 39.825546762636236], [50.09945606265317, 39.809181231354216], [49.73962419841186, 39.79082773795352], [49.411828538246574, 39.771531669765125], [49.112903926865414, 39.752361911025716], [48.83970937087893, 39.734370841690804], [48.588904283874484, 39.71853350278657], [48.356668874133405, 39.705685709856574], [48.138442903230484, 39.696496930069166], [47.928788827720375, 39.691510765787285], [47.72146742060195, 39.69125314972629], [47.509733380285766, 39.69636259878296], [47.28675426521679, 39.70767684238989], [47.04600044965878, 39.72623722199169], [46.781813896647854, 39.75318231004165], [46.492712354833024, 39.78930965557465], [46.184855426544104, 39.835465738965375], [45.86503420630595, 39.89276276812895], [45.538837356377435, 39.962281730100244], [45.21200592246691, 40.044861885428254], [44.89039410906893, 40.141018114396786], [44.579894027748956, 40.250840340305835], [44.2863040639904, 40.373881513270526], [44.01510021046895, 40.509060707169276], [43.77106667167757, 40.65464388574588], [43.5573813666099, 40.808763112256635], [43.370366831929395, 40.974976310843005], [43.19978703499116, 41.16174154090347], [43.03989035147257, 41.37464259245645], [42.89017131554361, 41.61576096267359], [42.7511200820247, 41.88610196752454], [42.62365020597018, 42.18565609185887], [42.50872885643805, 42.51356021345258], [42.407152726568974, 42.86832190155592], [42.319444200498616, 43.248033985620495], [42.24582700974895, 43.650546622095334], [42.18624364799504, 44.07360868195933], [42.14035445676405, 44.51535600053171], [42.10734654881238, 44.97451969017837], [42.08603878611024, 45.44965300153019], [42.07513610049457, 45.93894773346013], [42.07321631336564, 46.44063727405062], [42.07875468798453, 46.95302574990553], [42.09014469000815, 47.474493093635], [42.10571343571256, 48.00349684107231], [42.123732375123865, 48.53857250242879], [42.14242370747497, 49.07833364734001], [42.15996212021977, 49.62144475040475], [42.17447154025788, 50.165397240477475], [42.18495767634588, 50.706741455107064], [42.191765130134385, 51.24418512774656], [42.19539080264513, 51.77823355392378], [42.19633311592158, 52.309529725626255], [42.195087932397925, 52.83872712710596], [42.19214800176526, 53.36649008883457], [42.188002537163506, 53.893493960974475], [42.18313693587219, 54.420425112104915], [42.17803265227176, 54.94798076199131], [42.17316720647656, 55.47687081765356], [42.16901141663257, 56.00806709322738], [42.16585234165179, 56.54236613895102], [42.16363096566183, 57.079048315926485], [42.16225287388822, 57.616734168234814], [42.161623801682794, 58.153990576613765], [42.16164924347823, 58.68938858596864], [42.16223450333168, 59.22150310508622], [42.16328473271554, 59.74891275495821], [42.16470495814562, 60.270199862923256], [42.16640010020944, 60.78395061100724], [42.168274984636476, 61.288755359752834], [42.17023434508243, 61.78320918534581], [42.17218281608796, 62.265912690676565], [42.17402491292669, 62.735473184363606], [42.17566499228539, 63.19050637271495], [42.17700718301762, 63.6296387901238], [42.17795526785442, 64.05151132485021], [42.17841248147248, 64.45478441890259], [42.17828116045483, 64.83814590823015], [42.17746212047907, 65.20032317359897], [42.175853508516646, 65.54010261061117], [42.17331771769265, 65.85945697102157]]
    wx = [point[0]/K_SIZE for point in waypoints]
    wy = [point[1]/K_SIZE for point in waypoints]

    #left_lane = [[55.20794875342281, 35.27448377932513], [54.41200077143069, 35.30695012048921], [53.6994973087895, 35.32973278451455], [53.05369198332356, 35.34434617374535], [52.46995563559294, 35.35176073641368], [51.94352348534089, 35.352944721617924], [51.469438115878916, 35.348855260008904], [51.04247945768565, 35.34042476383173], [50.65708612827037, 35.328542623703854], [50.307279937694375, 35.3140345137871], [49.98661664339574, 35.2976458764601], [49.68819974536164, 35.28004191430688], [49.40480489639883, 35.26184083609993], [49.12915942550908, 35.24369385917416], [48.854389847740535, 35.22640687872313], [48.57458108875353, 35.21106424090563], [48.285298967621294, 35.199082988630906], [47.983863913457455, 35.19213189441894], [47.66919972248722, 35.19191507008499], [47.341242340112316, 35.19991145908137], [47.00010199166572, 35.21720157978887], [46.64528327084289, 35.24445872656409], [46.2745475937944, 35.28216173174648], [45.87931746481253, 35.331656328205064], [45.45324927161776, 35.395799297901505], [44.99775611322375, 35.477744386118395], [44.51673049611575, 35.58071462633185], [44.01376015987991, 35.70841650498975], [43.49220652653013, 35.86520526907044], [42.95535520393836, 36.05628607951973], [42.406679252140755, 36.28793804365331], [41.85029651678887, 36.567708021260174], [41.29170841989968, 36.90444909239871], [40.73962895448888, 37.30728213810211], [40.21572305213209, 37.77404841556637], [39.74445738388303, 38.28697971301097], [39.33389566558475, 38.829869763591645], [38.98144517365749, 39.393424459311106], [38.682521943220564, 39.970502054575995], [38.43169520892545, 40.55599046603325], [38.22342697741633, 41.14648482447686], [38.05252020717205, 41.739838877752334], [37.914326474681424, 42.3347358533025], [37.804794661660715, 42.93034432457531], [37.72043612551989, 43.526035435958946], [37.65828813675138, 44.12042943788908], [37.615451425478646, 44.711485825454254], [37.58888022548739, 45.298049615959194], [37.57571823961618, 45.87971176495256], [37.57336339157698, 46.455980049564076], [37.579418721360284, 47.02622076988066], [37.591650864164635, 47.589648613976586], [37.6079591539506, 48.14532306347987], [37.62635425717835, 48.69214765126249], [37.644945341894676, 49.22886979134787], [37.66193659970435, 49.75413508783306], [37.67569420948184, 50.26898879204658], [37.6855596795437, 50.778784214723295], [37.6919971426274, 51.28792595240343], [37.69544606865983, 51.7974714207315], [37.69634288552445, 52.30820264243259], [37.695129140323296, 52.82088065570283], [37.692252605644555, 53.33624480629904], [37.688168127178734, 53.855012400596586], [37.68333818233723, 54.37787870813712], [37.678233134329474, 54.905517296089904], [37.67333121489716, 55.43857635711526], [37.6691195807439, 55.977163825405434], [37.66591095107731, 56.519748410337705], [37.663657635570566, 57.063966803996976], [37.662261809378535, 57.608331338126], [37.66162534740354, 58.15146167115417], [37.66165060638612, 58.69196913317576], [37.66224032381785, 59.22845732652209], [37.66329754235989, 59.75952242911462], [37.66472555458655, 60.28375320618094], [37.666427864928686, 60.79973071357419], [37.6683081675333, 61.30602765011388], [37.67027034069487, 61.8012072833333], [37.67221846093807, 62.28382182735665], [37.67405684332149, 62.75241008485172], [37.675690120077576, 63.205494063107345], [37.677023379105115, 63.641574113242044], [37.67796240054794, 64.0591218786405], [37.6784140606586, 64.45656989519092], [37.67828703187694, 64.83229591089969], [37.67749302845373, 65.18459858413947], [37.67594910205193, 65.51165854361095], [37.67364381202733, 65.80528370570941]]
    bx1 = [point[0]/K_SIZE for point in left_lane]
    by1 = [point[1]/K_SIZE for point in left_lane]
    
    #right_lane = [[55.452051246577184, 41.269516220674866], [54.62989532023595, 41.30293496054599], [53.863168690817474, 41.327434531145926], [53.15958130197929, 41.34333784829438], [52.51472587464332, 41.351511781923264], [51.92433075054822, 41.35282540109815], [51.384316907864445, 41.348158893332126], [50.890867977595285, 41.33841716454403], [50.4405089037134, 41.32454814228036], [50.03018143763943, 41.30756347054325], [49.65729338341723, 41.288555025117994], [49.31970480254155, 41.26869492158454], [49.01560360368761, 41.24920226933431], [48.743226019335545, 41.23126316919635], [48.50040909591914, 41.21590904414105], [48.28403146926003, 41.20389286617356], [48.089490881766885, 41.19563491054859], [47.91043046580802, 41.19130372291007], [47.73888998664019, 41.19103250960673], [47.56589706034359, 41.19517964535016], [47.382305023067154, 41.20450192992356], [47.1795728425974, 41.22016338713422], [46.950902664265676, 41.243522502806705], [46.69717731817319, 41.27519409803117], [46.428724144852886, 41.31535455265333], [46.154126904000016, 41.36443556213246], [45.87953964313133, 41.42280409802304], [45.611421176662574, 41.490343678907756], [45.35645663658186, 41.5662890628389], [45.121406969019155, 41.64902509390121], [44.91284566794028, 41.73586266980959], [44.736701441695644, 41.82284493580565], [44.59751942227021, 41.9047088168616], [44.496632170650244, 41.97592343697481], [44.4219147585285, 42.041952275935216], [44.35156358536054, 42.119995483534304], [44.275221913435175, 42.22290020207805], [44.19308002950565, 42.35653979712775], [44.107319461626076, 42.524635271840715], [44.02096853831843, 42.72887796713407], [43.93716281611196, 42.96925200977782], [43.858696899701286, 43.24448290949046], [43.78781677577101, 43.55246669639316], [43.72617112577836, 43.890614054602004], [43.674846155486755, 44.25613309729279], [43.63437656343494, 44.64699818807926], [43.60464492325696, 45.06219764508641], [43.58509163965118, 45.500187463387185], [43.5749420541207, 45.958693056295985], [43.5731672872952, 46.43552301554614], [43.57853334352595, 46.92862740991382], [43.589642631955996, 47.4361079201878], [43.60496486296654, 47.956221433603126], [43.62285841443904, 48.487380786150894], [43.641583162668404, 49.028154932670724], [43.65930396039157, 49.57721463792865], [43.6740639838499, 50.130866723287774], [43.684757008613275, 50.68272720190165], [43.69168779263671, 51.229604852860945], [43.69537238064023, 51.771820931654545], [43.6963298593873, 52.309972086690806], [43.6950741964228, 52.84467595090701], [43.69211313380549, 53.37657184967974], [43.68794734049177, 53.90632114776711], [43.68306985371718, 54.43460724676085], [43.677965824919184, 54.96213525062512], [43.673112537003036, 55.489635637833], [43.668975361928794, 56.01836818250137], [43.66583280517662, 56.54990538182212], [43.66362207569225, 57.08407548656966], [43.662249895391454, 57.619535111604414], [43.66162328644255, 58.154833545100296], [43.6616487891756, 58.688528403566266], [43.66223256316963, 59.219185031274264], [43.66328046283409, 59.74537619690607], [43.664698092665304, 60.26568208183736], [43.66639084530302, 60.77869057681826], [43.668263923670864, 61.28299792963248], [43.670222346544946, 61.77720981934998], [43.672170934471254, 62.25994297844987], [43.67401426946176, 62.72982755086757], [43.67565661635466, 63.18551047591749], [43.677001784321796, 63.625660349084384], [43.677952890289916, 64.04897447358678], [43.6784119550771, 64.45418926013981], [43.678279203314126, 64.8400959073403], [43.677451817820845, 65.2055647034188], [43.675821644004884, 65.54958396627792], [43.67320901958109, 65.87751472612563]]
    bx2 = [point[0]/K_SIZE for point in right_lane]
    by2 = [point[1]/K_SIZE for point in right_lane]

    #C.obs = np.array([[39.174046199856555/K_SIZE, 62.74676445135568/K_SIZE],[42.07875468798453/K_SIZE, 46.95302574990553/K_SIZE]])
    for obstacle in map_data["obstacles"]:
        obstacle_dict[obstacle[0]] = [obstacle[1],obstacle[2]]
    obstacle_array = np.array(list(map(lambda o: Obstacle(obstacle_dict[o][0], obstacle_dict[o][1], (3.14/2),C.W,C.RF), obstacle_dict.items())))

    obs_x = [x for x, y in obstacle_array]
    obs_y = [y for x, y in obstacle_array]
    rx, ry, ryaw, rk, ref_path = get_reference_line(wx, wy)

    l0 = 0.0  # current lateral position [m]
    l0_v = 0.0  # current lateral speed [m/s]
    l0_a = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    s0_v = 1/5  # current speed [m/s]
    s0_a = 0.0

    rospy.init_node("lattice_planner", anonymous=True)
    pub_waypoints = rospy.Publisher("waypoints", WaypointList, queue_size=10)
    rospy.Subscriber("donkey/pose", PoseStamped, new_pose)
    rospy.Subscriber("obstacles", Obstacles, new_obstacles)

    while True:
        path = lattice_planner_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)

        if path is None:
            print("No feasible path found!!")
            pub_waypoints.publish(True, [])
            break

        l0 = path.l[1]
        l0_v = path.l_v[1]
        l0_a = path.l_a[1]
        s0 = path.s[1]
        s0_v = path.s_v[1]
        s0_a = path.s_a[1]

        if np.hypot(path.x[1], path.y[1] - 0) <= 1.5:
            print("Goal")
            pub_waypoints.publish(True, [])
            break

        waypoints = list(map(lambda i:[path.x[i],path.y[i]], range(0,len(path.x))))
        pub_waypoints.publish(True, waypoints)

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(rx, ry, linestyle='--', color='gray')
        plt.plot(bx1, by1, linewidth=1.5, color='k')
        plt.plot(bx2, by2, linewidth=1.5, color='k')
        plt.plot(path.x[1:], path.y[1:], linewidth='2', color='royalblue')
        draw.draw_car(path.x[1], path.y[1], path.yaw[1], 0.0, C)
        plt.title("[Stopping Mode]  v :" + str(s0_v * 3.6)[0:4] + " km/h")
        plt.axis("equal")
        plt.pause(0.0001)

    plt.pause(0.0001)
    plt.show()

    plt.plot(rx, ry, linestyle='--', color='gray')
    plt.plot(bx1, by1, linewidth=1.5, color='k')
    plt.plot(bx2, by2, linewidth=1.5, color='k')
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    try:
        main_Crusing()
        #main_Stopping()
    except rospy.ROSInterruptException:
        pass
    
