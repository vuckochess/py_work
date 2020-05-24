import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import Chessboard_manipulations as cbm
import Corner_model_keras as cmk
import Fen_string_manipulations as fen

Options = fen.Options

class ImageManipulation:
    def __init__(self, image, corner_coordinates, corner_pos):
        self.image = image
        self.coords = np.array(corner_coordinates)
        self.corner_pos = corner_pos
        self.scale = self.image.size[0]/cmk.PIC_WIDTH#/cmk.PIC_HEIGHT

    def copy(self):
        return ImageManipulation(self.image, self.coords, self.corner_pos)

    def scale_and_remove_color(self, num_copies, train):
        image_c = []
        rot_c = []

        rotated_image = None
        rotated_coord = None

        for _ in range(num_copies):
            if train:
                crop_image = self.copy()
                crop_image.random_cropping()
                scale = crop_image.scale
                # crop_image.display_image()
                rotated_image, rotated_coord = cbm.random_rotate_and_translate(crop_image.image, crop_image.coords)
            else:
                scale = self.scale
                rotated_image = self.image
                rotated_coord = np.copy(self.coords)

            rotated_image = rotated_image.resize((cmk.PIC_WIDTH, cmk.PIC_HEIGHT))
            rotated_image = rotated_image.convert('L')
            rotated_coord = (rotated_coord/scale).round().astype(int).flatten()

            np_image = np.array(rotated_image)
            ####################################################
            np_image = extend_picture(np_image, cmk.OFFSET)
            ####################################################
            image_c += [np_image]
            rot_c += [rotated_coord.tolist()]
            rotated_image.close()
        # if not train:
        #     x_coord = rotated_coord[0: :2]
        #     y_coord = rotated_coord[1: :2]
        #     draw = ImageDraw.Draw(rotated_image)
        #     cbm.draw_rect_from_arrays(draw, x_coord, y_coord, cbm.COLORS)
        #     rotated_image.show()
        # plt.clf()
        # plt.imshow(np.array(rotated_image))
        # plt.show()
        self.image.close()

        # image_copies = np.array(image_c)
        # rot_coords = np.array(rot_c)
        # return image_copies, rot_coords
        return image_c, rot_c


    def display_image(self):
        print('Display.')
        x_coord = self.coords[:, 0]
        y_coord = self.coords[:, 1]
        display = self.image.copy()
        draw = ImageDraw.Draw(display)
        cbm.draw_rect_from_arrays(draw, x_coord, y_coord, cbm.COLORS)
        display_nparray_image(np.array(display))
        display.close()

    def random_cropping(self):
        width, height = self.image.size
        x_coords = self.coords[:, 0]
        y_coords = self.coords[:, 1]
        cropping_box = self.generate_cropping_coords(width, height, x_coords, y_coords)
        self.image = self.image.crop(cropping_box)
        self.scale = self.image.size[0]/cmk.PIC_WIDTH
        # print('New size:', self.image.size)
        self.calculate_new_coords(cropping_box)


    def calculate_new_coords(self, cropping_box):
        # print('Old coords:', self.coords)
        # print('Cropping box:', cropping_box)
        self.coords[:, 0] = self.coords[:, 0] - cropping_box[0]
        self.coords[:, 1] = self.coords[:, 1] - cropping_box[1]
        # print('New coords:', self.coords)


    def generate_cropping_coords(self, width, height, x_coords, y_coords):
        coord_left = np.min(x_coords)
        coord_right = np.max(x_coords)
        coord_up = np.min(y_coords)
        coord_bottom = np.max(y_coords)

        SCALE_CONST = 0.90
        width_cut_max = (np.min((width-(coord_right-coord_left), (height-(coord_bottom-coord_up))*width/height)))*SCALE_CONST
        while True:
            start_cut = int (np.random.random()*coord_left)
            end_cut = int (np.random.random()*(width-coord_right))
            if start_cut + end_cut < width_cut_max:
                start_w = start_cut
                end_w = width-end_cut
                break

        height_leave = (end_w - start_w) * height/width
        if height_leave<coord_bottom-coord_up:
            print('cut left: {:d}'.format(start_w))
            print('cut right: {:d}'.format(width-end_w))
            print('ERROR!!!!!!!')
        height_cut = height - height_leave
        
        count = 0
        while True:
            count += 1
            partition = np.random.random()
            start_cut = int (partition*height_cut)
            end_cut = height_cut-start_cut
            if start_cut<coord_up and end_cut < height-coord_bottom:
                start_h = start_cut
                end_h = height-end_cut
                if count>20:
                    print('iteration:', count)
                break

        min_cut = np.max((height_cut - height + coord_bottom + 1, 0))
        max_cut = np.min((height_cut, coord_up))
        start_h = int (min_cut + np.random.random()*(max_cut-min_cut))
        end_h = int (height_leave + start_h)
        # print('width:', width, ', coord_left:', coord_left, ', coord_right:', coord_right)
        # print('start_w:', start_w, 'end_w:', end_w)
        # print('height:', height, ', coord_up:', coord_up, ', coord_bottom:', coord_bottom)
        # print('start_h:', start_h, 'end_h:', end_h)
        # print('New scale: {:.2f}, old scale: {:.2f}.\r\n\r\n'.format((end_w-start_w)/(end_h-start_h), width/height))
        # print(start_w, start_h, end_w, end_h)
        return start_w, start_h, end_w, end_h


def extend_picture(np_matrix, offset, double_offset=True):
    height, width = np_matrix.shape
    if double_offset:
        new_matrix = np.zeros(np.array((height, width)) + 2*offset)
        new_matrix[offset[0] : height+offset[0], offset[1] : width+offset[1]] = np.copy(np_matrix)
    else:
        new_matrix = np.zeros(np.array((height, width)) + offset)
        new_matrix[:height, :width] = np.copy(np_matrix)
    return new_matrix


def get_class_samples(np_matrix, class_matrix, height, width, offset):
    class_vector = class_matrix.ravel()
    class_samples = np.nonzero(class_vector).tolist()
    while True:
        rnd = int (np.random.random()*class_vector.size)
        if rnd not in class_samples:
            class_samples += [rnd]
            break
    class_h, _ = class_matrix.shape
    class_samples = [[x//class_h, x%class_h] for x in class_samples]
    samples = []
    for sample_order in class_samples:
        samples += get_samples(np_matrix, sample_order, height, width, offset)
    return samples


def get_samples(np_matrix, sample_order, height_d, width_d, offset):
    small_pic = []
    for sample_h, sample_w in sample_order:
        small_pic += [np.copy(np_matrix
                    [sample_h*height_d : sample_h*height_d+height_d+offset,
                    sample_w*width_d : sample_w*width_d+width_d+offset]
                    )]
    return small_pic

def display_nparray_image(np_array):
    plt.clf()
    plt.imshow(np_array, cmap='gray')
    plt.show()


def load_data(data_list, train=True):
    import os.path as path

    if train:
        pickle_file = Options.pickle_train_file
    else:
        pickle_file = Options.pickle_test_file
    if not path.exists(pickle_file):
        # pickle_data = create_pickle_file(data_list, pickle_file)
        pickle_data = create_image_data(data_list)
    else:
        print('Loading pickled pics.')
        pickle_data = pickle.load(open(pickle_file, 'rb'))
        print(len(pickle_data), 'pictures.')
    return pickle_data


def create_pickle_file(data_list, pickle_file):
    print('Creating pickle file.')
    multi_image_data = create_image_data(data_list)
    pickle.dump(multi_image_data, open(pickle_file, 'wb'))
    print('File', pickle_file, 'created.')
    return multi_image_data


def create_image_data(data_list):
    counter = 0
    multi_image_data = {}
    for file_name, coords, corner_pos in data_list:
        counter += 1
        if counter % 100 == 0:
            print('Counter: {:d}/{:d}'.format(counter, len(data_list)))
        image = Image.open(file_name)
        image_data = ImageManipulation(image, coords, corner_pos)
        multi_image_data[file_name] = [np.array(image_data.image), coords, corner_pos]
    return multi_image_data


def generate_data(data_list, num_of_copies, train=True):
    import time
    print('Start generating data.')
    start_t = time.time()
    x_ = []
    y_ = []
    pickle_data = load_data(data_list, train)

    counter = 0
    for file_name in pickle_data:
        np_array, coords, corner_pos = pickle_data[file_name]
        counter += 1
        if counter % 50 == 0:
            print('Counter: {:d}/{:d}'.format(counter*num_of_copies, len(data_list)*num_of_copies))
        image = Image.fromarray(np_array)
        image_data = ImageManipulation(image, coords, corner_pos)

        image_copies, new_coords = image_data.scale_and_remove_color(num_of_copies, train)
        x_ += image_copies
        y_ += coords_to_grid_categories(new_coords)
        # DISPLAY_COUNT = 15
        # if counter<DISPLAY_COUNT:
        #     print('Grid positions:\r\n{}'.format(y_[counter-1]))
        #     print('Coord array: {}'.format(new_coords))
        #     display_nparray_image(x_[counter-1])

    print('End generating data.')
    x_data = np.array(x_)
    y_data = np.array(y_)
    print(x_data.shape, y_data.shape)
    end_t = time.time()
    print(str(round(end_t-start_t)) + ' seconds spent.')

    return x_data, y_data


def coords_to_grid_categories(coords):
    all_coords = np.array(coords).reshape(-1, 4, 2)
    cat_array = []
    for np_coords in all_coords:
        categories = np.zeros((cmk.GRID_HEIGHT_DIM, cmk.GRID_WIDTH_DIM))
        for i in range(cmk.CAT_NUM-1):
            x_grid = np_coords[i][0]//cmk.GRID_SQUARE_SIZE
            y_grid = np_coords[i][1]//cmk.GRID_SQUARE_SIZE
            # if x_grid>=0 or y_grid>=0 x_grid>=cmk.GRID_WIDTH_DIM or y_grid>=cmk.GRID_HEIGHT_DIM:
            #     print('Index out of range: {}, {}, should be between (0, {}) and (0, {}).'.format(x_grid, y_grid, cmk.GRID_WIDTH_DIM-1, cmk.GRID_HEIGHT_DIM-1))
            #     print('Coord array: {}'.format(coords))
            if x_grid>=0 and x_grid<cmk.GRID_WIDTH_DIM and y_grid>=0 and y_grid<cmk.GRID_HEIGHT_DIM:
                categories[y_grid, x_grid] = i+1
        cat_array += [categories]
    return cat_array


def fetch_data_list(train=True):
    if train:
        json_file_name = Options.train_file
        num_of_copies = Options.num_of_copies
    else:
        json_file_name = Options.test_file
        num_of_copies = 1
    data_dict = fen.fetch_dict_from_json(json_file_name)
    data_list = []
    for file_name in data_dict:
        data_list.append([file_name, data_dict[file_name][0], data_dict[file_name][1]])
        # print(file_name, data_dict[file_name][0])
    return data_list, num_of_copies


def run_corner_network():
    data_list, num = fetch_data_list(True)
    x_train, y_train = generate_data(data_list, num, True)
    data_list, num = fetch_data_list(False)
    x_test, y_test = generate_data(data_list, num, False)
    predictor = cmk.CornerPredictor()#x_train, y_train, x_test, y_test)
    predictor.load_data(x_train, y_train, x_test, y_test)

    predictor.show_predicted_data()
    predictor.train_model()
    predictor.show_predicted_data()
    predictor.evaluate_model()

# if __name__ == '__main__':
#     a = np.arange(12).reshape(3,4)
#     print(extend_picture(a, 2))
#     print(extend_picture(a, 2, True))
