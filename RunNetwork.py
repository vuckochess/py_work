import numpy as np

import Corner_network as cn
import Corner_model_keras as cmk
import Piece_network as pn
import Piece_model_keras as pmk
import Fen_string_manipulations as fen

Options = fen.Options

def test_data():
    data_list, num = cn.fetch_data_list(False)
    x_test, y_test = cn.generate_data(data_list, num, False)
    predictor = cmk.CornerPredictor()
    predictor.load_data(x_test, y_test, x_test, y_test)
    y_pred = predictor.model.predict(predictor.x_test)
    corner_pred = (y_pred + cmk.AVG_CORNER_DATA[2]) * 10 / 3
    # corner_pred = (y_pred + cmk.AVG_CORNER_DATA[2]) * 5
    corner_pred = corner_pred.round().astype(int).reshape(-1, 4, 2).tolist()

    fen_dict = fen.fetch_dict_from_json(Options.board_string_file)
    corner_data, num = pn.fetch_corner_data(False)
    new_corner_data = []

    new_corners = []
    old_corners = []
    for i in range(len(corner_data)):
        new_corner_data.append([corner_data[i][0], corner_pred[i], corner_data[i][2]])
        old_corners += [np.array(corner_data[i][1]).flatten()]
        new_corners += [np.array(corner_pred[i]).flatten()]
        # print("Old corners:", old_corners[i])
        # print("New corners:", new_corners[i])
    print('MSE:', np.mean(np.square(np.array(old_corners) - np.array(new_corners))))

    # x_image, y_image = pn.image_croping(new_corner_data, 1, fen_dict, False)
    x_image, y_image = pn.image_croping(corner_data, 1, fen_dict, False)
    image_pred = pmk.PiecePredictor()
    image_pred.load_data(x_image, y_image, x_image, y_image)
    y_pred = image_pred.model.predict(image_pred.x_test)
    y_ = np.argmax(y_pred, axis=1)

    
    # strings_array = fen.conv_num_array_to_board_strings(y_)
    # for expected, predicted in zip(data_list, strings_array):
    #     if fen_dict[expected[0]] != predicted:
    #         print("Expected string: " + fen_dict[expected[0]] + ", predicted: " + predicted)
    # for i in range(len(y_pred)):
    #     if y_pred[i][y_[i]]<0.99:
    #         print("Error greater than 1%:", y_pred[i].round(3))
    print("Accuracy {:.2f}%".format(np.mean(np.equal(y_, y_image.flatten()).astype(int))*100))



def main():
    cn.run_corner_network()
    # pn.run_piece_network()
    # test_data()


if __name__ == '__main__':
    main()
