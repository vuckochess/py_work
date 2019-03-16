import numpy as np
import time

import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import Fen_string_manipulations as fen
import Chessboard_manipulations as cbm
import Piece_model_keras as pmk

import Corner_network as cn

Options = fen.Options

def print_error_data(errored):
    pieces_dir = './Chess positions/Pieces/'
    for i in range(len(errored)):
        exp_c = fen.PIECE_NUM_DICT[errored[i][0]]
        pred_c = fen.PIECE_NUM_DICT[errored[i][1]]
        # print('Expected: ' + exp_c + '\r\nPredicted:' + pred_c, "\r\nErrored: " + errored[i][3].round(2))
        new_file_name = pieces_dir + 'Error_' + str(i) + '_exp_' + exp_c + '__pred_' + pred_c + '.JPG'
        plt.imsave(new_file_name, errored[i][2], cmap='gray')


def image_croping(corner_data, num_of_copies, fen_dict, train=True):
    print("Processing images:")

    start = time.time()

    # PART_NUM = 8
    PART_NUM = 1
    PART_LEN = np.ceil(len(corner_data)/PART_NUM).astype(int)
    print(PART_LEN)
    x_data = []
    y_data = []
    parts = PART_NUM
    # for i in range(10):
    #     print(corner_data[i][0])
    if train:
        parts = 1
    for i in range(parts):
        print('Part', i)
        # image_data = cn.load_data(corner_data, train)
        # part = corner_data
        part = corner_data[i*PART_LEN:(i+1)*PART_LEN]
        image_data = cn.create_image_data(part)
        file_names = list(image_data)
        x_d, y_d = generate_augmented_data(image_data, file_names, corner_data, fen_dict, num_of_copies, train)
        x_data += x_d
        y_data += y_d

    print('Data loaded in {:.2f} seconds.'.format(time.time()-start))

    # parts = []
    # start = time.time()
    # import multiprocessing as mp
    # pool = mp.Pool(processes=PART_NUM)
    # output_data = [pool.apply(generate_augmented_data,
    #     args = (image_data, parts[i], corner_data, fen_dict, num_of_copies, train))
    #     for i in range(PART_NUM)]
    # print('Now unzip!')
    # x_data, y_data = zip(*output_data)

    # output = mp.Queue()
    # processes = [mp.Process(target=generate_augmented_data,
    #     args = (image_data, parts[i], corner_data, fen_dict, num_of_copies, train, output))
    #     for i in range(2)]
    # # Run processes
    # for p in processes:
    #     p.start()
    # # Exit the completed processes
    # for p in processes:
    #     p.join()
    # output_data = [output.get() for p in processes]
    # print('Output obtained.')

    # for i in range(PART_NUM):
    #     print('Part', i)
    #     x_d, y_d = generate_augmented_data(image_data, parts[i], corner_data, fen_dict, num_of_copies, train)
    #     x_data += x_d
    #     y_data += y_d

    print('End generating data. {:.2f} s spent.'.format(time.time()-start))
    x_data = np.array(x_data).reshape(-1, pmk.PIC_WIDTH, pmk.PIC_HEIGHT)
    y_data = np.array(y_data).reshape(-1, 1)

    print('x_data.shape:', x_data.shape)
    print('y_data.shape:', y_data.shape)
    # plt.clf()
    # print(y_data[100])
    # plt.imshow(x_data[100], cmap='gray')
    # plt.show()

    return x_data, y_data

def generate_augmented_data(pickle_data, files_part, corner_data, fen_dict, num_of_copies, train):

    x_data = []
    y_data = []
    counter = 0
    for file_name in files_part:
        np_array, coords, corner_pos = pickle_data[file_name]
        counter += 1
        if counter % 30 == 0:
            print("Processed " + str(counter) + "/" + str(len(corner_data)))
        image = Image.fromarray(np_array)
        fen_string = fen_dict[file_name]
        fen_string = fen.rotate_board_multiple(fen_string, (corner_pos+3)%4)

        for _ in range(num_of_copies):
            if train:
                image, coords = cbm.random_rotate_and_translate(image, coords)

            x_image_list, y_image_list = cbm.create_subimages(image, coords, fen_string, train)
            x_data += x_image_list
            y_data += y_image_list
        image.close()
    # print('Out of generate_augmented_data()')
    return (x_data, y_data)

def fetch_corner_data(train=True):
    if train:
        json_file_name = Options.train_file
        num_of_copies = Options.num_of_copies
    else:
        json_file_name = Options.test_file
        num_of_copies = 1
    data_dict = fen.fetch_dict_from_json(json_file_name)
    data_list = []
    for key in data_dict:
        data_list.append([key, data_dict[key][0], data_dict[key][1]])
    return data_list, num_of_copies


def run_piece_network():
    piece_net = pmk.PiecePredictor()

    fen_dict = fen.fetch_dict_from_json(Options.board_string_file)
    corner_data, num = fetch_corner_data(True)
    x_train, y_train = image_croping(corner_data, num, fen_dict, True)

    corner_data, num = fetch_corner_data(False)
    x_test, y_test = image_croping(corner_data, num, fen_dict, False)

    piece_net.load_data(x_train, y_train, x_test, y_test)
    piece_net.train_model()
    piece_net.evaluate_model()

    errored = piece_net.show_predicted_data()
    print_error_data(errored)

