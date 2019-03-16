import numpy as np
import random
import Fen_string_manipulations as fen

import matplotlib.pyplot as plt

COLORS = ['red', 'yellow', 'green', 'pink']

def random_symmetric(num):
    return (random.random()*2-1) * num


def get_coord_matrix(array):
    coord_matrix = np.zeros((9,9,2))
    coord_matrix[8,0] = array[0]
    coord_matrix[8,8] = array[1]
    coord_matrix[0,8] = array[2]
    coord_matrix[0,0] = array[3]

    retrieve_coordinates(coord_matrix, 0, 0, 8)
    return coord_matrix


def retrieve_coordinates(coord_matrix, left, bottom, dim):

    point1 = Point(*coord_matrix[left+dim, bottom])
    point2 = Point(*coord_matrix[left+dim, bottom+dim])
    point3 = Point(*coord_matrix[left, bottom+dim])
    point4 = Point(*coord_matrix[left, bottom])

    diag1 = Line(point1, point3)
    diag2 = Line(point2, point4)

    intercept = diag1.intersection(diag2)
    coord_matrix[left+dim//2, bottom+dim//2] = intercept.to_array()

    line_a = Line(point1, point4)
    line_h = Line(point2, point3)
    line_1 = Line(point1, point2)
    line_8 = Line(point3, point4)

    line_med = line_1.medium_slope(line_8, intercept)
    coord_matrix[left+dim//2, bottom] = line_a.intersection(line_med).to_array()
    coord_matrix[left+dim//2, bottom+dim] = line_h.intersection(line_med).to_array()

    line_upright = line_a.medium_slope(line_h, intercept)
    coord_matrix[left, bottom+dim//2] = line_8.intersection(line_upright).to_array()
    coord_matrix[left+dim, bottom+dim//2] = line_1.intersection(line_upright).to_array()

    dim = dim//2
    if dim == 1:
        return
    retrieve_coordinates(coord_matrix, left, bottom, dim)
    retrieve_coordinates(coord_matrix, left+dim, bottom, dim)
    retrieve_coordinates(coord_matrix, left, bottom+dim, dim)
    retrieve_coordinates(coord_matrix, left+dim, bottom+dim, dim)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_string(self):
        return "[" + str(self.x) + ", " + str(self.y) + "]"
    
    def to_array(self):
        return [self.x, self.y]

class Line:
    def __init__(self, point1=None, point2=None):
        self.slope = 0
        self.y_intercept = 0
        if point1 == None or point2 == None:
            return
        if point1.x == point2.x:
            self.slope = 0
            self.y_intercept = point1.x
        else:
            self.slope = (point1.y - point2.y)/(point1.x - point2.x)
            self.y_intercept = point1.y - self.slope * point1.x

    def intersection(self, line):
        if self.slope == line.slope:
            print("Lines are parallel:", self.slope, self.y_intercept, ";", line.slope, line.y_intercept)
            return None
        x = -(self.y_intercept - line.y_intercept)/(self.slope-line.slope)
        y = self.slope*x + self.y_intercept
        return Point(x,y)

    def medium_slope(self, line, point):
        if self.slope == line.slope:
            y_intercept = point.y - self.slope * point.x
            med_line = Line()
            med_line.slope = self.slope
            med_line.y_intercept = y_intercept
            return med_line
        intersect = self.intersection(line)
        return Line(point, intersect)


def center_coordinates(coord_matrix):
    center_matrix = np.zeros((8,8,2))
    scale_matrix = np.ones((8,8))
    # print(scale_matrix)
    row_dim = len(coord_matrix)
    column_dim = len(coord_matrix[0])
    for i in range(row_dim-1):
        for j in range(column_dim-1):
            center_matrix[i,j] = square_center(coord_matrix[i,j], coord_matrix[i,j+1], coord_matrix[i+1,j+1], coord_matrix[i+1,j])
            scale_matrix[i,j] = (coord_matrix[i+1,j][0] - coord_matrix[i+1,j+1][0]) / (coord_matrix[row_dim-1,j][0] - coord_matrix[row_dim-1,j+1][0])
    return center_matrix, scale_matrix


def square_center(left_up, right_up, right_down, left_down):
    point1 = Point(*left_up)
    point2 = Point(*right_up)
    point3 = Point(*right_down)
    point4 = Point(*left_down)

    diag1 = Line(point1, point3)
    diag2 = Line(point2, point4)
    intercept = diag1.intersection(diag2)
    return intercept.to_array()


def random_rotate_and_translate(image, coord, max_rot=5, max_transl=(30,20)):
    angle = random_symmetric(max_rot)
    translate = [random_symmetric(max_transl[0]), random_symmetric(max_transl[1])]
    center = np.array(image.size)/2

    rot_image = image.rotate(angle, translate=translate)
    rot_coord = rotate_points(center, angle, coord, translate)
    return rot_image, rot_coord



def rotate_points(center, angle, points_array, translate):
    np_coord = (np.array(points_array) - center) * [1, -1]
    x = np_coord[:, 0]
    y = np_coord[:, 1]
    radius = np.sqrt(x**2 + y**2)
    radian = np.arctan(y/x)
    # print(x)
    radian += (1 - np.sign(x)) * np.radians(90)

    new_radian = radian + np.radians(angle)
    x_rot = radius * np.cos(new_radian)
    y_rot = radius * np.sin(new_radian)
    rot_points = np.stack((x_rot, y_rot), axis=1) * \
        [1, -1] + center + translate
    return rot_points


def create_subimages(image, coords, fen_string, train):
    fen_to_array = fen.convert_fen_to_array(fen_string)
    img_scaling = ImageScaling(image, coords)

    x_data = []
    y_data = []
    # x_data = np.empty((64, 202, 146))
    # y_data = np.empty((64, 1))

    for row in range(8):
        for column in range(8):
            croped_image = img_scaling.extract_image(row, column, train)
            x_data += [np.array(croped_image)]
            
            index = fen_to_array[column + row*8]
            y_data += [index]
            # x_data[column + row*8] = np.array(croped_image)
            # y_data[column + row*8] = [index]

    return x_data, y_data


def make_lower_resolution(image, scale):
    width, height = image.size
    image = image.resize((round(width/scale), round(height/scale))).resize((width, height))
    return image


def plot_gray_image(image):
    plt.clf()
    plt.imshow(image, cmap='gray')
    plt.show()


def draw_rect_from_arrays(draw, x_coord, y_coord, colors, option=0):
    if option==0:
        for x, y, color in zip(x_coord, y_coord, colors):
            if x>=0 and y>=0:
                x0, y0, x1, y1 = x-3, y-3, x+3, y+3
                draw.rectangle([x0, y0, x1, y1], fill=color)
    else:
        x0, y0, x1, y1 = x_coord[0], y_coord[0], x_coord[1], y_coord[1] 
        if x0>=0 and y0>=0 and x1>=0 and y1>=0:
            draw.rectangle([x0, y0, x1, y1], outline=colors[1])




class ImageScaling:
    SCALE = 5/8
    UP_SCALE_RATIO = 7/4
    WINDOW_SIZE = np.array([-SCALE, -UP_SCALE_RATIO*SCALE, SCALE, SCALE])
    ANGLE_CHANGE = [3, 0]
    COORD_CHANGE = [6, 0]


    def __init__(self, image, coords):
        self.image = image
        self.coord_matrix = get_coord_matrix(coords)
        self.center_spots, self.scale_matrix = center_coordinates(self.coord_matrix)
        self.basis = np.sqrt(np.square(self.center_spots[7,0,0] - self.center_spots[7,1,0]) +
            np.square(self.center_spots[7,0,1] - self.center_spots[7,1,1]))


    def extract_image(self, row, column, train):
        if train:
            option = 0
        else:
            option = 1
        i_scale = ImageScaling

        local_scale = self.scale_matrix[row, column]*self.basis
        basis_box = (local_scale*2) * i_scale.WINDOW_SIZE
        square_center = self.center_spots[row, column]
        box = np.append(square_center, square_center)
        box = (box + basis_box).round()
        croped_image = self.image.crop(box)

        x_diff = self.coord_matrix[7,3,0] - square_center[0]
        y_diff = self.image.size[1] * 13 / 5 - square_center[1]

        angle = -np.degrees(np.arctan(x_diff/y_diff)) + random_symmetric(i_scale.ANGLE_CHANGE[option])
        translate = [-np.tan(np.radians(angle)) * local_scale / 2 + random_symmetric(i_scale.COORD_CHANGE[option]),
                    random_symmetric(i_scale.COORD_CHANGE[option])]
        croped_image = croped_image.rotate(angle, translate=translate)
        width, height = (box[2] - box[0]) / 4, (box[3] - box[1]) / (2 * (i_scale.UP_SCALE_RATIO+1))
        small_box = np.array([width, height * i_scale.UP_SCALE_RATIO,
                        width * 3, height * (2 * i_scale.UP_SCALE_RATIO + 1)]).round()
        croped_image = croped_image.crop(small_box)

        croped_image = croped_image.resize((146, 202)).convert('L')
        return croped_image
