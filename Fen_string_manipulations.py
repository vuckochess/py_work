import json
import random
import numpy as np

TABLE_CORNERS_ALL = './Snapshots/snapshots_table_corners.json'

class Options:
    option = 1
    if option:
        train_file = './Snapshots/snapshots_train.json'
        test_file = './Snapshots/snapshots_test.json'
        board_string_file = './Snapshots/snapshots_pieces.json'
        num_of_copies = 1
        pickle_train_file = './Pickles/pickled_snapshots_train.p'
        pickle_test_file = './Pickles/pickled_snapshots_test.p'
    else:
        train_file = './Chess positions/table_corners_train.json'
        test_file = './Chess positions/table_corners_test.json'
        board_string_file = './Chess positions/file_dict_all.json'
        num_of_copies = 10
        pickle_train_file = './Pickles/pickled_pics_train.p'
        pickle_test_file = './Pickles/pickled_pics_test.p'



def make_char_dict():
    char_dict = {}
    piece_num_dict = {}
    for i, c in enumerate(CHARS):
        char_dict[c] = i
        piece_num_dict[i] = c
    return char_dict, piece_num_dict

DIR_NAME = './generated_data/'
CHARS = '0KQRBNPkqrbnp'
CHARS_LENGTH = len(CHARS)
CHAR_DICT, PIECE_NUM_DICT = make_char_dict()

def convert_fen_to_array(fen_string):
    y_array = [0]*64
    for row in range(8):
        for column in range(8):
            square_order = column + row*8
            piece_char = fen_string[square_order]
            y_array[square_order]=CHAR_DICT[piece_char]
    return y_array


def convert_array_to_string(board_array):
    board_string = ""
    for field in board_array:
        board_string += PIECE_NUM_DICT[field]
    return board_string


def conv_num_array_to_board_strings(num_array):
    num_array = num_array.reshape(-1, 64)
    board_strings_array = []
    for row in num_array:
        board_strings_array += [convert_array_to_string(row)]
    return board_strings_array


def fetch_dict_from_json(json_file_name):
        json_file = open(json_file_name, 'r')
        json_data = json_file.read()
        json_file.close()
        return json.loads(json_data)


def dump_dict_to_json_file(json_dict, json_file_name):
    json_data = json.dumps(json_dict, indent=4, sort_keys=True)
    json_file = open(json_file_name, 'w')
    json_file.write(json_data)
    json_file.close()


def separate_train_and_test_to_json_files(all_json, train_json, test_json, train_percent=80):
    json_dict = fetch_dict_from_json(all_json)
    train_dict, test_dict = dict_to_train_and_test(json_dict, int(len(json_dict)*train_percent/100))

    dump_dict_to_json_file(train_dict, train_json)
    dump_dict_to_json_file(test_dict, test_json)


def dict_to_train_and_test(dict_input, train_quantity):
    values = list(dict_input.values())
    # values = (np.array(values)*12/5).round().astype(int).tolist()
    list_input = [[x, y] for x, y in zip(dict_input.keys(), values)]
    random.shuffle(list_input)
    train_list = list_input[:train_quantity]
    test_list = list_input[train_quantity:]
    train_dict = {x: y for [x, y] in train_list}
    test_dict = {x: y for [x, y] in test_list}
    return train_dict, test_dict


def rotate_board_string(board_string):
    rot_string = ""
    for i in range(8):
        for j in range(8):
            rot_string += board_string[j*8 + (7-i)]
    return rot_string


def printout_board(board_str):
    printout = ""
    for i in range(8):
        printout += board_str[i*8:i*8+8] + "\r\n"
    return printout


def rotate_board_multiple(board_string, rot_num):
    rot_string = board_string
    for _ in range(rot_num):
        rot_string = rotate_board_string(rot_string)
    return rot_string


