import tkinter as tk
import tkinter.font
from PIL import Image, ImageTk, ImageDraw
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import json

def scale_and_remove_color(picture_file_name, new_file_name):
    image = Image.open(picture_file_name)
    image_copy = image.copy()
    image_copy = image_copy.resize((384, 216))
    image_copy = image_copy.convert('L')

    image_copy.save(new_file_name)

def insert_png_into_picture(picture_file_name, png_file_name, new_file_name):
    image = Image.open(picture_file_name)
    image_copy = image.copy()
    image_copy = image_copy.resize((768, 432))
    game_diagram = Image.open(png_file_name)
    insert_picture_into_corner(image_copy, game_diagram, new_file_name)

def insert_picture_into_corner(bg_picture, fg_picture, new_file_name):
    position = ((bg_picture.width-fg_picture.width)//2, (bg_picture.height-fg_picture.height)//2)
    bg_picture.paste(fg_picture, position, fg_picture)
    # Image.alpha_composite(bg_picture, fg_picture)
    # bg_picture = Image.blend(bg_picture, fg_picture, alpha=0.5)
    bg_picture.save(new_file_name)

def test_insert_piece():
    bg_picture = Image.open('./Chess positions/positions/DSC00516.JPG', 'r')
    fg_picture = Image.open('./Chess positions/positions/White_King.png', 'r')
    new_file_name = './Chess positions/positions/White King_inserted.JPG'
    insert_picture_into_corner(bg_picture, fg_picture, new_file_name)

def insert_kings_into_picture(coord_matrix, scale_matrix):
    bg_picture = Image.open('./Chess positions/positions/DSC00516.JPG', 'r')
    fg_picture = Image.open('./Chess positions/positions/White_King.png', 'r')
    new_file_name = './Chess positions/positions/DSC00516_wigh_Kings.JPG'

    width, height = fg_picture.size
    for row_c, row_s in zip(coord_matrix, scale_matrix):
        for elem_c, elem_s in zip(row_c, row_s):
            width_s, height_s = int(width * elem_s), int(height * elem_s)
            scaled_fg_picture = fg_picture.resize((int(width_s), int(height_s)))
            bg_picture.paste(scaled_fg_picture, (int(elem_c[0])-width_s*5//9, int(elem_c[1])-height_s*2//3), scaled_fg_picture)
    bg_picture.save(new_file_name)


def get_coord_matrix(array):
    coord_matrix = np.zeros((9,9,2))
    coord_matrix[8,0] = array[0]
    coord_matrix[8,8] = array[1]
    coord_matrix[0,8] = array[2]
    coord_matrix[0,0] = array[3]

    retrieve_coordinates(coord_matrix, 0, 0, 8)
    return coord_matrix

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

def list_files():
    file_dict = dict()
    png_files = os.listdir(MakePngFile.DIR_NAME)
    for file_name in png_files:
        sub_place = file_name.find('.jpg')
        if sub_place>=8:
            file_num = int(file_name[8:sub_place])
            file_dict[file_num] = file_name
    json_file_name = 'file_dict.json'

    json_data = json.dumps(file_dict, indent=4)
    json_file = open(json_file_name, 'w')
    json_file.write(json_data)


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


class MakePngFile:
    BORDER = 2
    SQUARE_DIM = 80
    ROW_DIM = 8
    COLUMN_DIM = 8
    DIR_NAME = './generated_data/'

    def __init__(self):
        self.read_board_coord()
        # self.generate_data_check()
        # self.create_png_files()

    def read_board_coord(self):
        json_file_name = './Chess positions/coordinates.json'
        json_file = open(json_file_name, 'r')
        json_data = json_file.read()
        json_file.close()

        file_dict = json.loads(json_data)
        for img in file_dict:
            array = file_dict[img]
            coord_matrix = get_coord_matrix(array)
            center_spots, scale_matrix = center_coordinates(coord_matrix)
            insert_kings_into_picture(center_spots, scale_matrix)
            
    def generate_data_check(self):
        MakePngFile.DIR_NAME = './Chess positions/'

        file_dict = dict()
        fen_file = open(MakePngFile.DIR_NAME + 'fen_data.txt', 'r')
        POSITIONS_DIR = MakePngFile.DIR_NAME + 'positions/'
        png_files = os.listdir(POSITIONS_DIR)

        for file_name in png_files:
            check_jpg_file = file_name.find('.JPG')
            if check_jpg_file<0:
                print("Not jpg:", file_name)
                continue

            board_string = fen_file.readline()
            if len(board_string) <=0:
                print("The end.")
                break
            board_string = board_string[0:64]
            file_dict[file_name] = board_string
            png_file_name = MakePngFile.DIR_NAME + 'Generated files/' + file_name
            self.generate_png_files(board_string, png_file_name)
            new_file_name = MakePngFile.DIR_NAME + 'Generated files/Grayscale/' + file_name
            scale_and_remove_color(POSITIONS_DIR + file_name, new_file_name)
            # insert_png_into_picture(POSITIONS_DIR + file_name, png_file_name, new_file_name)

        print(len(file_dict))
        json_file_name = MakePngFile.DIR_NAME + 'file_dict.json'

        json_data = json.dumps(file_dict, indent=4, sort_keys=True)
        json_file = open(json_file_name, 'w')
        json_file.write(json_data)
        json_file.close()


    def create_png_files(self):
        fen_file = open('fen_data.txt', 'r')
        file_num = 0
        filename = ''

        file_dict = dict()
        while True:
            file_num += 1
            board_string = fen_file.readline()
            board_string = board_string[0:64]
            if len(board_string) <=0 or file_num>1000:
                break
            else:
                filename = MakePngFile.DIR_NAME + 'test_img' + str(file_num) + '.jpg'
                self.generate_png_files(board_string, filename)
                file_dict[filename] = board_string
        # img = mpimg.imread(filename)
        # print(img.shape, img.dtype)

        json_file_name = 'file_dict.json'
        json_data = json.dumps(file_dict, indent=4)
        json_file = open(json_file_name, 'w')
        json_file.write(json_data)
        json_file.close()
        # plt.imshow(img, cmap="gray")
        # plt.imsave('./generated_data/test_img' + str(file_num) + '.jpg', img, cmap='gray')
        # plt.show()

    def generate_png_files(self, board_string, filename):
        big_weight = MakePngFile.SQUARE_DIM * MakePngFile.ROW_DIM + MakePngFile.BORDER
        big_height = MakePngFile.SQUARE_DIM * MakePngFile.COLUMN_DIM + MakePngFile.BORDER
        self.img = Image.new('L', (big_height, big_weight), color = 'white')
        self.draw_board = ImageDraw.Draw(self.img)
        self.draw_chess_board()
        self.add_pieces(board_string)
        self.img.thumbnail((128, 128), Image.ANTIALIAS)
        self.img.save(filename)


    def add_pieces(self, board_string):

        for pos in range(MakePngFile.ROW_DIM * MakePngFile.COLUMN_DIM):
            self.draw_piece(pos//8, pos%8, board_string[pos])

        # os.startfile(filename)


    def fill_square(self, pos_row, pos_col, color):
        allign_0 = MakePngFile.BORDER
        allign_1 = MakePngFile.BORDER + MakePngFile.SQUARE_DIM
        self.draw_board.rectangle((MakePngFile.SQUARE_DIM*pos_col + allign_0, MakePngFile.SQUARE_DIM*pos_row + allign_0,
                                MakePngFile.SQUARE_DIM*pos_col + allign_1, MakePngFile.SQUARE_DIM*pos_row + allign_1), fill=color)

    def draw_piece(self, pos_row, pos_col, piece):
        if piece == '0':
            return

        piece_dict = {'r': "BlackRook.png", 'n':"BlackKnight.png", 'b': "BlackBishop.png", 'q': "BlackQueen.png", 'k': "BlackKing.png", 'p': "BlackPawn.png",
            'R': "WhiteRook.png", 'N':"WhiteKnight.png", 'B': "WhiteBishop.png", 'Q': "WhiteQueen.png", 'K': "WhiteKing.png", 'P': "WhitePawn.png" }

        dim = MakePngFile.SQUARE_DIM
        piece_img = Image.open('./Pieces_png/' + piece_dict[piece])
        self.img.paste(piece_img, (dim*pos_col + MakePngFile.BORDER, dim*pos_row + MakePngFile.BORDER), piece_img)


    def draw_chess_board(self):

        for i in range(MakePngFile.ROW_DIM + 1):
            self.draw_board.line((MakePngFile.BORDER, i * MakePngFile.SQUARE_DIM + MakePngFile.BORDER, MakePngFile.COLUMN_DIM *
                               MakePngFile.SQUARE_DIM + MakePngFile.BORDER, i*MakePngFile.SQUARE_DIM + MakePngFile.BORDER))
        for i in range(MakePngFile.COLUMN_DIM + 1):
            self.draw_board.line((i * MakePngFile.SQUARE_DIM + MakePngFile.BORDER, MakePngFile.BORDER, i *
                               MakePngFile.SQUARE_DIM + MakePngFile.BORDER, MakePngFile.ROW_DIM*MakePngFile.SQUARE_DIM + MakePngFile.BORDER))
        for row in range(MakePngFile.ROW_DIM):
            for column in range(MakePngFile.COLUMN_DIM):
                if (row + column) % 2 == 0:
                    self.fill_square(row, column, 'white')
                else:
                    self.fill_square(row, column, 'grey')


def main():

    MakePngFile()


if __name__ == '__main__':
    main()
