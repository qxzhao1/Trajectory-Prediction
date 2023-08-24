import glob
from params import *
import json
import numpy
import math

#Features of raw data : [ 0: Frame number, 1: Vehicle ID, 2: Speed, 3: X position, 4: Y position, 5: Z position ]
def loadData(path):
    files = glob.glob(path+"vehicles*.json")
    vehicles, traffic_lights, speed_limiters = [], [], []
    for file in files:
        with open(file) as infile:
            jsonfile = json.load(infile)
        vehicles.append(jsonfile)
    with open(path+"traffic_lights.json") as infile:
        traffic_lights = json.load(infile)
    with open(path+"speed_limiter.json") as infile:
        speed_limiters = json.load(infile)
    return traffic_lights, vehicles, speed_limiters

#1 : Brake lights on, 0: Brake lights off
#Basically if speed decreases from frame -> frame -1, or if speed is minimal/stopping, then brake lights are on
def braking_lights(vehicles):
    for vehicle in vehicles:
        vehicle[0].append(0)
        for frame in range(1,len(vehicle)):
            if vehicle[frame][2] < vehicle[frame-1][2] or vehicle[frame][2] < speed_threshold :
                vehicle[frame].append(1)
            else:
                vehicle[frame].append(0)

#-1: "no lane", otherwise, lane numbers are inserted
#If the car is in the boundary of our hard-coded lane boundaries, then it is "in the lane"
def lane_numbering(vehicles, boundary):
    for vehicle in vehicles:
        for feature_per_vehicle in vehicle:
            add_lane_for_vehicle(vehicle, feature_per_vehicle, boundary)

def add_lane_for_vehicle(vehicle, feature_per_vehicle, boundary):
    for lane_num in boundary.keys():
        for lane_boundary in boundary[lane_num][1]:
            if feature_per_vehicle[3] >= min(lane_boundary[0:2]) and feature_per_vehicle[4] >= min(lane_boundary[2:4]) and feature_per_vehicle[3] <= max(lane_boundary[0:2]) and feature_per_vehicle[4] <= max(lane_boundary[2:4]):
                feature_per_vehicle.append(lane_num)
                break
        if len(feature_per_vehicle) == 8:
            break
    if len(feature_per_vehicle) != 8:
        feature_per_vehicle.append(-1)

#Blinkers: has two functions: blinking for lane change and for turning
#-1: left blinker,  0: no blinker, 1: right blinker
#If the lane changes from a valid lane number to another valid lane number: add_blink_lane_change()
#If the lane changes from a no lane (-1) to a valid lane number: add_blink_turn()
def blinkers(vehicles):
    for vehicle in vehicles:
        vehicle[0].append(0)
        for frame in range(1,len(vehicle)):
                if vehicle[frame][-1] != vehicle[frame-1][-2] and vehicle[frame-1][-2] != -1 and vehicle[frame][-1] != -1:
                    add_blink_lane_change(vehicle, frame)
                elif vehicle[frame][-1] != vehicle[frame-1][-2] and vehicle[frame-1][-2] == -1:
                    add_blink_turn(vehicle, frame)
                else:
                    vehicle[frame].append(0)

#Inserting blinkers for lane change
#Lanes are hardcoded where inner lanes are always smaller than outermost lanes
#idx -> 0: X-direction, 1: Y-direction.    Multiplier-> 1: right/up, -1:left/down
#threshold is the position which we want to update the blinkers until. This is determined by adding/subtracting our current position by the lane_threshold*multiplier
#This feature does not overwrite existing blinkers that are within the lane_threshold
def add_blink_lane_change(vehicle, frame):
    if vehicle[frame][-1] > vehicle[frame-1][-2] :
        blink = 1
    else:
        blink = -1
    idx, multiplier = find_direction(vehicle, frame-1)
    threshold = vehicle[frame][idx+3] - lane_threshold * multiplier
    while frame > 0 and vehicle[frame][idx+3] * multiplier >=  threshold * multiplier:
        if len(vehicle[frame]) == 9 and vehicle[frame][-1] == 0:
            l = vehicle[frame].pop(-1)
        if len(vehicle[frame]) != 9:
            vehicle[frame].append(blink)
        frame = frame - 1

#Inserting blinkers for turn
#change_threshold: Amount with which we determine if a car is traveling in a straight line or if it has turned. Any change in X or Y that is within this threshold has not turned.
#end_turn_frame is the first frame with a valid lane number, we then backtrack through all "no lanes" (-1) to find the previous lane to get start_turn_frame
#If the change in X or Y between start_turn_frame and end_turn_frame are <= change_threshold, then the car has gone in a straight line, hence blinkers = 0
#Else, it will find the blinker associated with the turn, and insert.
#Corner case: If the car is initialized in the intersection (No lane area), then blinkers will not go on.
#This feature does not overwrite existing blinkers in the intersection
def add_blink_turn(vehicle, frame):
    end_turn_frame = frame
    while frame > 0 and vehicle[frame-1][-2] == -1:
        frame = frame - 1
    if frame > 0:
        start_turn_frame, blink = evaluate_turn(vehicle, end_turn_frame, frame)
    else:
        start_turn_frame = 0
        blink = 0
    update_blink(vehicle, start_turn_frame, end_turn_frame, blink)

def evaluate_turn(vehicle, end_turn_frame, frame):
    start_turn_frame = frame
    if abs(vehicle[end_turn_frame][3] - vehicle[start_turn_frame][3]) <= change_threshold or abs(vehicle[end_turn_frame][4] - vehicle[start_turn_frame][4]) <= change_threshold:
        blink = 0
        vehicle[end_turn_frame].append(blink)
    else:
        idx, multiplier= find_direction(vehicle, start_turn_frame)
        blink = find_blink(vehicle, start_turn_frame, end_turn_frame, idx)
        threshold = vehicle[start_turn_frame][idx+3] - turn_threshold * multiplier
        while start_turn_frame > 0 and vehicle[start_turn_frame][idx+3] * multiplier >=  threshold * multiplier:
            start_turn_frame = start_turn_frame - 1
    return start_turn_frame, blink

def update_blink(vehicle, start_turn_frame, end_turn_frame, blink):
    for frame in range(start_turn_frame, end_turn_frame+1):
        if len(vehicle[frame]) == 9 and vehicle[frame][-1] == 0:
            l = vehicle[frame].pop(-1)
        if len(vehicle[frame]) != 9:
            vehicle[frame].append(blink)

#Used to replace "no lanes" with the next valid lane number the car is at.
#Deletes frames that end with "no lane", and has no next valid lane number.
#Applies to intersection and corners.
def merge_lanes(vehicles):
    for vehicle in vehicles:
        frame = 0
        while frame < len(vehicle):
            if vehicle[frame][7] == -1:
                frame = merge_lane(vehicle, frame)
            else:
                frame = frame + 1

def merge_lane(vehicle, frame):
    start_frame = frame
    while frame < len(vehicle) and vehicle[frame][7] == -1:
        frame = frame + 1
    if frame < len(vehicle):
        end_frame = frame
        lane_num = vehicle[frame][7]
        for change_lane_frames in range(start_frame, end_frame):
            vehicle[change_lane_frames][7] = lane_num
    else:
        for change_lane_frames in range(start_frame, len(vehicle)):
            vehicle.pop(-1)
    return frame

#Traffic light information is hard-coded to the lane boundaries. Hence we just update the traffic light status (0: green, 1: yellow, 2: red) and Euclidean distance.
#Based on what lane number the car is on.
def add_traffic_lights(vehicles, traffic_lights, boundary):
    for vehicle in vehicles:
        for frame in range(len(vehicle)):
            lane_num = vehicle[frame][7]
            tlight = traffic_lights[boundary[lane_num][0]]
            vehicle[frame].append(tlight[frame][5])
            vehicle[frame].append(numpy.linalg.norm(numpy.asarray(vehicle[frame][3:5])-numpy.asarray(tlight[frame][2:4])))

#Finds the direction at which the car is traveling.
#Direction: 0 is X, 1 is Y.  Multiplier: -1 is going left/down, 1 is right/up
#First we find the earliest frame on the same lane, and take the X2-X1, and Y2-Y1 to see which is larger, and this tells us the Direction
#To tell the left/right, up/down trajectory of our car, so we take the delta/abs(delta) = multiplier
def find_direction(vehicle,frame):
    tempFrame = frame
    while tempFrame > 0 and vehicle[tempFrame][7] == vehicle[frame][7]:
        tempFrame = tempFrame - 1
    x2_x1 = vehicle[frame][3] - vehicle[tempFrame][3]
    y2_y1 = vehicle[frame][4] - vehicle[tempFrame][4]
    direction, multiplier = 0, 0
    if(abs(x2_x1) > abs(y2_y1)):
        direction = 0
        multiplier = (0.000000001 + x2_x1) / abs(0.000000001 + x2_x1)
    else:
        direction = 1
        multiplier =  (0.000000001 + y2_y1) / abs(0.000000001 + y2_y1)

    return direction, multiplier

#         |
#   -x,+y |  +x,+y
#         |
# ------------------
#         |
#   -x,-y |  +x,-y
#         |
#Finds the correct blinker to turn on
#Based off concept that assuming the car is traveling in the X-direction, and that the change in X and Y are both positive/negative, then the car has turned right, else, it has turned left. Reversed concept for if the car is traveling in the Y-direction
def find_blink(vehicle, start_frame, end_frame, idx):
    x2_x1 = vehicle[end_frame][3] - vehicle[start_frame][3]
    y2_y1 = vehicle[end_frame][4] - vehicle[start_frame][4]

    if idx == 0:
        if((x2_x1 < 0 and y2_y1 < 0) or (x2_x1 > 0 and y2_y1 > 0)):
            return 1    #turned right
        return -1       #turned left
    else:
        if((x2_x1 < 0 and y2_y1 < 0) or (x2_x1 > 0 and y2_y1 > 0)):
            return -1    #turned left
        return 1       #turned right

def add_speed_threshold(vehicles, speed_limiters):
    for vehicle in vehicles:
        vehicle[0].append(0.0)
        start_frame = 0
        for frame in range(1, len(vehicle)):
            for speed_segment in speed_limiters:
                check_and_update_speed(vehicle[frame], speed_segment)
                if (len(vehicle[frame]) == 12):
                    print ("Added speed: ", vehicle[frame][11], "  to frame: ", frame)
                    if(start_frame == 0):
                        print("going to updating_first_speed")
                        update_first_speeds(vehicle, start_frame, frame)
                        start_frame = 1
                    break
            if len(vehicle[frame]) < 12:
                vehicle[frame].append(vehicle[frame-1][-1])
                print("Did not add in frame: ", frame, ", so appending old speed of: ", vehicle[frame-1][-1])

def update_first_speeds(vehicle, start_frame, end_frame):
    speeds = [8.333333969116211, 11.111111640930176, 16.666667938232422]
    smallestDelta = 9999.99
    chosen = 0
    for i in range(len(speeds)):
        currentDelta = abs(vehicle[end_frame-1][2] - speeds[i])
        if currentDelta <= smallestDelta:
            chosen = i
            smallestDelta = currentDelta

    while(start_frame<end_frame):
        vehicle[start_frame][-1] = speeds[chosen]
        print("Recursive updating frame: ", start_frame, " with value: ", speeds[chosen])
        start_frame+=1


def check_and_update_speed(features, speed_segment):
    direction = speed_segment[0]
    if (features[4-direction] <= speed_segment[2-direction]+1) and (features[4-direction] >= speed_segment[2-direction]-1) and (features[3+direction] <= speed_segment[1+direction] + speed_sign_threshold) and (features[3+direction] >= speed_segment[1+direction] - speed_sign_threshold):
        features.append(speed_segment[3])


#Features of data : [ 0: Frame number, 1: Vehicle ID, 2: Speed, 3: X position, 4: Y position, 5: Z position, 6: Brake Lights, 7: Lane Number, 8: Blinkers, 9: Traffic Light Status, 10: Distance from traffic light, 11: Speed Limiter for this segment]
def save_data(vehicles):
    for i in range(len(vehicles)):
        print ("Shape of ", i , " is: ", len(vehicles[i]), "x", len(vehicles[i][0]))
        numpy.savetxt(processed_data_dir+'p3vehicle'+str(i)+'.txt',vehicles[i],delimiter=",")

def check_speed_limiter(speed_limiters):
    for i in range(len(speed_limiters)):
        print("entry: ", i, " has length: ", len(speed_limiters[i]))

traffic_lights, vehicles, speed_limiters = loadData(raw_data_dir)
braking_lights(vehicles)
lane_numbering(vehicles, boundary)
blinkers(vehicles)
merge_lanes(vehicles)
add_traffic_lights(vehicles, traffic_lights, boundary)
add_speed_threshold(vehicles, speed_limiters)
save_data(vehicles)
