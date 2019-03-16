import tkinter as tk
import tkinter.font
from PIL import Image, ImageDraw
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import json
import Chessboard_manipulations as cbm
import Fen_string_manipulations as fen

piece_dict = {'r': "BlackRook",
              'n': "BlackKnight",
              'b': "BlackBishop",
              'q': "BlackQueen",
              'k': "BlackKing",
              'p': "BlackPawn",
              'R': "WhiteRook",
              'N': "WhiteKnight",
              'B': "WhiteBishop",
              'Q': "WhiteQueen",
              'K': "WhiteKing",
              'P': "WhitePawn"}


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
    position = ((bg_picture.width-fg_picture.width)//2,
                (bg_picture.height-fg_picture.height)//2)
    bg_picture.paste(fg_picture, position, fg_picture)
    # Image.alpha_composite(bg_picture, fg_picture)
    # bg_picture = Image.blend(bg_picture, fg_picture, alpha=0.5)
    bg_picture.save(new_file_name)


def test_insert_piece():
    bg_picture = Image.open('./Chess positions/positions/DSC00516.JPG', 'r')
    fg_picture = Image.open(
        './Chess positions/Night positions/Black_King_Night.png', 'r')
    new_file_name = './Chess positions/positions/White King_inserted.JPG'
    insert_picture_into_corner(bg_picture, fg_picture, new_file_name)


def insert_kings_into_picture(img_name, coord_matrix, scale_matrix):
    bg_picture = Image.open(img_name, 'r')
    pieces_dir = './Chess positions/Night positions/'
    piece = 'Black_Pawn'
    fg_picture = Image.open(pieces_dir + piece + '_Night.png', 'r')
    new_file_name = ('./Chess positions/Night positions/'
                     + 'Position_with_Bishops.JPG')

    width, height = fg_picture.size
    for row_c, row_s in zip(coord_matrix, scale_matrix):
        for elem_c, elem_s in zip(row_c, row_s):
            width_s, height_s = int(width*elem_s), int(height*elem_s)
            scaled_fg_picture = fg_picture.resize(
                (int(width_s), int(height_s)))
            bg_picture.paste(
                scaled_fg_picture,
                (int(elem_c[0])-width_s*5//9, int(elem_c[1])-height_s*2//3),
                scaled_fg_picture)
    bg_picture.save(new_file_name)


def list_files():
    file_dict = dict()
    png_files = os.listdir(MakePngFile.DIR_NAME)
    for file_name in png_files:
        sub_place = file_name.find('.jpg')
        if sub_place >= 8:
            file_num = int(file_name[8:sub_place])
            file_dict[file_num] = file_name
    json_file_name = 'file_dict.json'

    json_data = json.dumps(file_dict, indent=4)
    json_file = open(json_file_name, 'w')
    json_file.write(json_data)


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
        file_dict = fen.fetch_dict_from_json(json_file_name)

        for img in file_dict:
            array = file_dict[img]
            coord_matrix = cbm.get_coord_matrix(array)
            center_spots, scale_matrix = cbm.center_coordinates(coord_matrix)
            insert_kings_into_picture(img, center_spots, scale_matrix)

    def generate_data_check(self):
        MakePngFile.DIR_NAME = './Chess positions/'

        file_dict = dict()
        fen_file = open(MakePngFile.DIR_NAME + 'fen_data.txt', 'r')
        POSITIONS_DIR = MakePngFile.DIR_NAME + 'positions/'
        png_files = os.listdir(POSITIONS_DIR)

        for file_name in png_files:
            check_jpg_file = file_name.find('.JPG')
            if check_jpg_file < 0:
                print("Not jpg:", file_name)
                continue

            board_string = fen_file.readline()
            if len(board_string) <= 0:
                print("The end.")
                break
            board_string = board_string[0:64]
            file_dict[file_name] = board_string
            png_file_name = (MakePngFile.DIR_NAME
                             + 'Generated files/'
                             + file_name)
            self.generate_png_files(board_string, png_file_name)
            new_file_name = (MakePngFile.DIR_NAME
                             + 'Generated files/Grayscale/'
                             + file_name)
            scale_and_remove_color(POSITIONS_DIR + file_name,
                                   new_file_name)
            # insert_png_into_picture(POSITIONS_DIR + file_name,
            #                       png_file_name, new_file_name)

        print(len(file_dict))
        json_file_name = MakePngFile.DIR_NAME + 'file_dict.json'
        fen.dump_dict_to_json_file(file_dict, json_file_name)

    def create_png_files(self):
        fen_file = open('fen_data.txt', 'r')
        file_num = 0
        filename = ''

        file_dict = dict()
        while True:
            file_num += 1
            board_string = fen_file.readline()
            board_string = board_string[0:64]
            if len(board_string) <= 0 or file_num > 1000:
                break
            else:
                filename = MakePngFile.DIR_NAME + \
                    'test_img' + str(file_num) + '.jpg'
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
        # plt.imsave('./generated_data/test_img' + str(file_num) + '.jpg',
        #               img, cmap='gray')
        # plt.show()

    def generate_png_files(self, board_string, filename):
        big_weight = (MakePngFile.SQUARE_DIM*MakePngFile.ROW_DIM
                      + MakePngFile.BORDER)
        big_height = (MakePngFile.SQUARE_DIM*MakePngFile.COLUMN_DIM
                      + MakePngFile.BORDER)
        self.img = Image.new('L', (big_height, big_weight), color='white')
        self.draw_board = ImageDraw.Draw(self.img)
        self.draw_chess_board()
        self.add_pieces(board_string)
        self.img.thumbnail((128, 128), Image.ANTIALIAS)
        self.img.save(filename)

    def add_pieces(self, board_string):
        for pos in range(MakePngFile.ROW_DIM*MakePngFile.COLUMN_DIM):
            self.draw_piece(pos//8, pos % 8, board_string[pos])

    def fill_square(self, pos_row, pos_col, color):
        allign_0 = MakePngFile.BORDER
        allign_1 = MakePngFile.BORDER + MakePngFile.SQUARE_DIM
        self.draw_board.rectangle((
            MakePngFile.SQUARE_DIM*pos_col + allign_0,
            MakePngFile.SQUARE_DIM*pos_row + allign_0,
            MakePngFile.SQUARE_DIM*pos_col + allign_1,
            MakePngFile.SQUARE_DIM*pos_row + allign_1),
            fill=color)

    def draw_piece(self, pos_row, pos_col, piece):
        if piece == '0':
            return

        dim = MakePngFile.SQUARE_DIM
        piece_img = Image.open('./Pieces_png/' + piece_dict[piece] + '.png')
        self.img.paste(piece_img,
                       (dim*pos_col + MakePngFile.BORDER,
                        dim*pos_row + MakePngFile.BORDER),
                       piece_img)

    def draw_chess_board(self):

        for i in range(MakePngFile.ROW_DIM + 1):
            self.draw_board.line((
                MakePngFile.BORDER,
                i*MakePngFile.SQUARE_DIM + MakePngFile.BORDER,
                MakePngFile.COLUMN_DIM*MakePngFile.SQUARE_DIM + MakePngFile.BORDER,
                i*MakePngFile.SQUARE_DIM + MakePngFile.BORDER))
        for i in range(MakePngFile.COLUMN_DIM + 1):
            self.draw_board.line((
                i*MakePngFile.SQUARE_DIM + MakePngFile.BORDER,
                MakePngFile.BORDER,
                i*MakePngFile.SQUARE_DIM + MakePngFile.BORDER,
                MakePngFile.ROW_DIM*MakePngFile.SQUARE_DIM + MakePngFile.BORDER))
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
