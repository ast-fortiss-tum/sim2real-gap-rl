#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import CubicSpline
import json
import os



class For_convertion_utils():
    def __init__(self,size_factor,x_map_shift,y_map_shift,angle_shift):
        self.size_factor=size_factor
        self.x_map_shift=x_map_shift
        self.y_map_shift=y_map_shift
        self.angle_shift=angle_shift
    
    def real2sim_xyzypr(self,pose,orientation):
        x=pose[0]*self.size_factor+self.x_map_shift
        y=pose[1]*self.size_factor+self.y_map_shift
        angle=-orientation[2]+self.angle_shift
        return [x,y,angle]

    def sim2real_xyzypr(self,pose,orientation):
        x=(pose[0]-self.x_map_shift)/self.size_factor
        y=(pose[2]-self.x_map_shift)/self.size_factor
        angle=-(orientation[1]-self.angle_shift)
        return [x,y,angle]
    
    def real2sim_xyp(self,pose):
        x,y,p=pose
        x=x*self.size_factor+self.x_map_shift
        y=y*self.size_factor+self.y_map_shift
        angle=-p+self.angle_shift
        return [x,y,angle]

    def sim2real_xyp(self,pose):
        x,y,p=pose
        x=(x-self.x_map_shift)/self.size_factor
        y=(y-self.y_map_shift)/self.size_factor
        angle=-(p-self.angle_shift)
        return [x,y,angle]


class Road_shapes_utils():    
    def __init__(self,scale_1, scale_2, x_map_shift, y_map_shift, size_factor,angle_shift):
        self.scale_1, self.scale_2, self.x_map_shift, self.y_map_shift, self.size_factor,self.angle_shift=scale_1, scale_2, x_map_shift, y_map_shift, size_factor,angle_shift
        self.real_initial_waypoint = None
        self.real_initial_angle = None
        self.sim_initial_waypoint = None
        self.sim_initial_angle = None
    

    def generate_road_margins(self,road_points, offset):
        left_margins = []
        right_margins = []

        num_points = len(road_points)

        # Calculate the direction vectors for each road segment
        direction_vectors = []
        for i in range(num_points - 1):
            dx = road_points[i + 1][0] - road_points[i][0]
            dy = road_points[i + 1][1] - road_points[i][1]
            mag = np.sqrt(dx ** 2 + dy ** 2)
            direction_vectors.append((dx / mag, dy / mag))

        # Average neighboring direction vectors to get smoother normals
        averaged_directions = []
        for i in range(num_points - 1):
            if i == 0:
                averaged_directions.append(direction_vectors[0])
            elif i == num_points - 2:
                averaged_directions.append(direction_vectors[-1])
            else:
                averaged_directions.append(((direction_vectors[i][0] + direction_vectors[i - 1][0]) / 2,
                                            (direction_vectors[i][1] + direction_vectors[i - 1][1]) / 2))

        # Calculate normals and generate margins
        for i in range(num_points - 1):
            dx, dy = averaged_directions[i]
            nx = -dy
            ny = dx

            left_x = road_points[i][0] + offset * nx
            left_y = road_points[i][1] + offset * ny
            right_x = road_points[i][0] - offset * nx
            right_y = road_points[i][1] - offset * ny

            left_margins.append([left_x, left_y])
            right_margins.append([right_x, right_y])

        return left_margins, right_margins

    def create_non_closed_road(self,intermediate_points,start_angle,start_offset):
            for_conversions=For_convertion_utils(self.size_factor,self.x_map_shift,self.y_map_shift,self.angle_shift)
            starting_x_real, starting_y_real, starting_angle_real = intermediate_points[0][0],intermediate_points[0][1], start_angle
            if start_offset:
                initial_pose_real=[starting_x_real+start_offset[0], starting_y_real+start_offset[1], starting_angle_real]
            else:
                initial_pose_real=[starting_x_real, starting_y_real, starting_angle_real]
            initial_pose_sim=for_conversions.real2sim_xyp(initial_pose_real)
            scaled_vertices = []
            for point in intermediate_points:
                x,y,_= for_conversions.real2sim_xyp([point[0],point[1],0])
                scaled_vertices.append([x, y])

            # Interpolate a smooth road using cubic spline
            scaled_vertices = np.array(scaled_vertices)
            cs = CubicSpline(np.arange(len(scaled_vertices)), scaled_vertices)

            # Generate interpolated points along the smooth road
            num_interpolation_points = len(scaled_vertices) * 10  # Adjust as needed
            interpolated_points = cs(np.linspace(0, len(scaled_vertices) - 1, num_interpolation_points))

            # Format the road segments
            road_segments = []
            road_points =[]
            for point in interpolated_points:
                x, z = point
                road_points.append([x,z])
                y = 0.6  # Assuming constant y value
                road_segment = f"{x:.2f},{y:.2f},{z:.2f}"
                road_segments.append(road_segment)
            self.real_initial_waypoint=[initial_pose_real[0],initial_pose_real[1]]
            self.real_initial_angle=initial_pose_real[2]
            self.sim_initial_waypoint=[initial_pose_sim[0],initial_pose_sim[1]]
            self.sim_initial_angle=initial_pose_sim[2]
            return initial_pose_sim,initial_pose_real,road_points, '@'.join(road_segments)
    
    def create_circular_road(self,radius_real):
        radius_sim=radius_real*self.size_factor

        center_x_sim=self.x_map_shift
        center_y_sim=self.y_map_shift

        road_definition,road_points = self.create_circle(center_x_sim,center_y_sim,radius_sim, 100)
        for_conversions=For_convertion_utils(self.size_factor,self.x_map_shift,self.y_map_shift,self.angle_shift)
        
        initial_pose_sim=self.x_map_shift+radius_sim,self.y_map_shift,0
        initial_pose_real=for_conversions.sim2real_xyp(initial_pose_sim)
        self.real_initial_waypoint=[initial_pose_real[0],initial_pose_real[1]]
        self.real_initial_angle=initial_pose_real[2]

        self.sim_initial_waypoint=[initial_pose_sim[0],initial_pose_sim[1]]
        self.sim_initial_angle=initial_pose_sim[2]
        return initial_pose_sim,initial_pose_real,road_points, road_definition


    def create_circle(self,start_x, start_z, radius, num_segments):
        road_segments = []
        road_segments_complete = []
        interpolated_points = []
        theta_values = np.linspace(0, 2*np.pi, num_segments)
        for theta in theta_values:
            x = start_x + radius * np.cos(theta)
            z = start_z + radius * np.sin(theta)
            interpolated_points.append([x,z])
            y = 0.6
            road_segment = f"{x:.2f},{y:.2f},{z:.2f}"
            road_segments.append(road_segment)
        # road_segments_complete.append(road_segments[-1])
        for road_segment in road_segments:
            road_segments_complete.append(road_segment)
        return '@'.join(road_segments_complete),interpolated_points
    


SCALE_1=2.08
SCALE_2=3.2
SIZE_FACTOR=7.33
X_MAP_SHIFT=48
Y_MAP_SHIFT=50
ANGLE_SHIFT=0

#choose road
ROAD_TYPE = 'Guericke'

#########
ROAD_TYPE = 'Straight_line'


road_generation = Road_shapes_utils(SCALE_1, SCALE_2, X_MAP_SHIFT, Y_MAP_SHIFT, SIZE_FACTOR, ANGLE_SHIFT)

#generate road elements
if ROAD_TYPE=='Straight_line':
        road_points=[[0,-1.8],[0,10.6]]
        start_angle=0
        start_offset_real=[0.22,-0.3]
        initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_non_closed_road(road_points,start_angle,start_offset_real)
if ROAD_TYPE=='Guericke':
        road_points=[[1.,-1.6],[0.2,-1.6],[-0.2,-1.6],[-0.75,-1.4],[-1,-0.8],[-1,0],[-1,0.8],[-1,1.6],[-1,2.2]]
        start_angle=+90
        start_offset_real=None
        initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_non_closed_road(road_points,start_angle,start_offset_real)
if ROAD_TYPE=='Guericke_straight' or ROAD_TYPE=='Guericke_straight_obs':
        road_points=[[0,-1.8],[0,-1.6],[0,-1.3],[0,-0.9],[0,-0.6],[0,0],[0,0.8],[0,1.6],[0,2.2],[0,2.6]]
        start_angle=0
        start_offset_real=[0.22,-0.3]
        initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_non_closed_road(road_points,start_angle,start_offset_real)
if ROAD_TYPE=='Circuit':
        road_points = [[0.8, -1.0],  # Starting point
    [0.8, 1.6],   # Move up
    [-0.8, 1.6],  # Move left
    [-0.8, -1.0], # Move down
    [0.8, -1.0] ]   # Move right back to the starting point]
        start_angle=0
        start_offset_real=[0.22,-0.3]
        initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_non_closed_road(road_points,start_angle,start_offset_real)
if ROAD_TYPE=='Circuit_2':
        road_points = [
    [0.0, -1.8],  # Starting point shifted back
    [0.4, -0.8],  # First gentle curve up and right
    [0.0, 0.2],  # Curve back down and left
    [-0.4, 1.0],  # Curve back down and right
    [0.0, 1.8],    # Curve back down and left
    [0.4, 2.4]   # Ending point
]
        start_angle=0
        start_offset_real=[0.22,-0.3]
        initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_non_closed_road(road_points,start_angle,start_offset_real)

elif ROAD_TYPE=='Circle' or ROAD_TYPE=='Circle_obs':
    radius_real=1.3
    initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_circular_road(radius_real)

#generate lanes
left_lane, right_lane = road_generation.generate_road_margins(sim_waypoint_list, 1.5)
left_margin, right_margin = road_generation.generate_road_margins(sim_waypoint_list, 3)
#TODO: OJO! en el codigo original primero se generan los margenes con offset 1.5 y luego los márgenes en status tienen offset 3



#Generate obstacles
obstacles = []
#obstacles.append(["Wall",42,47,0,0,-30,0])
#obstacles.append(["cone",51,60,0,0,-90,0])



#load map elements into json file
map = {
    "initial_pose_sim": initial_pose_sim,
    "initial_pose_real": initial_pose_real,
    "sim_waypoint_list": sim_waypoint_list,
    "road_definition": road_definition,
    "left_lane": left_lane,
    "right_lane": right_lane,
    "left_margin": left_margin,
    "right_margin": right_margin,
    "obstacles": obstacles
}

with open("map_straight.json", "w") as f:
    f.write(json.dumps(map))

print(os.getcwd())
