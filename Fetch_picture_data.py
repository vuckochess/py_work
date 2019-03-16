import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

import Fen_string_manipulations as fen
import Chessboard_manipulations as cbm


class MyCanvas(tk.Canvas):
    WIDTH = 800
    HEIGHT = 450
    OPTION = 0
    OPTIONS = [[1, 2, 3, 4], [1,2]]
    NUM_STR = [['First', 'Second', 'Third', 'Fourth'], ['First', 'Second']]

    def __init__(self, master):
        self.master = master
        super().__init__(self.master, width=MyCanvas.WIDTH +
                         100, height=MyCanvas.HEIGHT, highlightthickness=0)
        self.length = len(MyCanvas.OPTIONS[MyCanvas.OPTION])
        self.snapshots_json = {}
        self.initialize_coords()
        self.fetch_data()
        self.create_labels()
        self.add_next_button()
        self.add_option_menu()
        self.scale = 1.0
        super().bind("<Button-1>", self.mouse_clicked)
        super().pack()

    def fetch_data(self):
        self.snapshots_json = fen.fetch_dict_from_json('./Snapshots/snapshots_table_corners.json')
        # self.jpg_files = []
        # for file in snapshots_json:
        #     if file.find('Petrosian-Geller')>=0:
        #         self.jpg_files.append(file)
        self.jpg_files = list(self.snapshots_json)
        self.jpg_files.sort()
        self.current_file_num = 0

    def create_image(self):
        self.initialize_coords()
        self.open_picture()
        self.peek_corners()
        self.show_picture()
        self.pack()

    def peek_corners(self):
        file_name = self.jpg_files[self.current_file_num]
        if file_name in self.snapshots_json:
            coords = np.array(self.snapshots_json[file_name][0])
            self.current_corner = (self.snapshots_json[file_name][1]-1)%self.length
            self.set_option()
            self.x_coord = (coords[:,0]/self.scale).astype(int).tolist()
            self.y_coord = (coords[:,1]/self.scale).astype(int).tolist()
            print(self.x_coord, self.y_coord)
            self.display_coord()
        else:
            print('File data for', file_name, 'not found.')

    def initialize_coords(self):
        self.x_coord = [-1] * self.length
        self.y_coord = [-1] * self.length
        self.current_corner = 0

    def open_picture(self):
        self.image = Image.open(self.jpg_files[self.current_file_num])
        self.scale = self.image.size[0]/MyCanvas.WIDTH
        self.image.thumbnail((MyCanvas.WIDTH, MyCanvas.HEIGHT))
        self.draw = ImageDraw.Draw(self.image)

    def create_labels(self):
        self.string_vars = []

        for i in range(self.length):
            self.string_vars.append(tk.StringVar())
            self.label = tk.Label(
                self.master, textvariable=self.string_vars[i], bg=cbm.COLORS[i])
            self.label.pack()
            self.label.place(x=MyCanvas.WIDTH, y=70*i +
                             10, width=100, height=50)
        self.display_coord()

    def add_next_button(self):
        self.next_button = tk.Button(
            self.master, text="Next", command=self.show_next_picture,
            bg='blue', fg='white')
        self.next_button.place(x=MyCanvas.WIDTH,y=70*self.length+10,
            width=100, height=50)

    def add_option_menu(self):
        self.option_var = tk.StringVar(self.master)
        self.option_var.set(MyCanvas.OPTIONS[MyCanvas.OPTION][1])

        self.option = tk.OptionMenu(
            self.master, self.option_var, *MyCanvas.OPTIONS[MyCanvas.OPTION],
            command=self.change_option_value)
        self.option.pack()
        self.option.place(x=MyCanvas.WIDTH, y=70*5+10,
            width=100, height=50)

    def change_option_value(self, event):
        self.current_corner = int(event)-1
        self.set_option()

    def set_option(self):
        self.option_var.set(MyCanvas.OPTIONS[MyCanvas.OPTION][self.current_corner])

    def increment_option_num(self):
        self.current_corner = (self.current_corner + 1) % self.length
        self.set_option()

    def mouse_clicked(self, event):
        self.x_coord[self.current_corner] = event.x
        self.y_coord[self.current_corner] = event.y
        self.increment_option_num()
        self.display_coord()

    def display_coord(self):
        
        append_text = ' corner:\r\n'

        for i in range(self.length):
            text = MyCanvas.NUM_STR[MyCanvas.OPTION][i] + append_text
            if self.x_coord[i] >= 0:
                text += 'x: ' + \
                    str(self.x_coord[i]) + ', y: ' + str(self.y_coord[i])
            else:
                text += 'x:     , y:    '
            self.string_vars[i].set(text)
        self.draw_rectangles()

    def show_picture(self):
        self.img = ImageTk.PhotoImage(self.image)
        super().create_image(MyCanvas.WIDTH//2, MyCanvas.HEIGHT//2, image=self.img)

    def show_next_picture(self):
        if self.current_file_num >= len(self.jpg_files):
            return
        printout = '"' + self.jpg_files[self.current_file_num] + '": [['
        next_pair = ''
        for x, y in zip(self.x_coord, self.y_coord):
            if x < 0:
                return
            if len(next_pair) > 0:
                printout += next_pair + ', '
            next_pair = '[' + str(int(x*self.scale)) + ', ' + str(int(y*self.scale)) + ']'
        printout += next_pair + ']'
        if self.length == 4:
            printout +=  ', ' + str(self.current_corner+1)
        printout += ']'
        print(printout)
        self.current_file_num += 1
        self.create_image()

    def draw_rectangles(self):

        self.open_picture()
        cbm.draw_rect_from_arrays(self.draw, self.x_coord,
                              self.y_coord, cbm.COLORS, MyCanvas.OPTION)
        self.show_picture()


def board_manipulations():
    master = tk.Tk()
    canvas = MyCanvas(master)
    canvas.create_image()
    master.mainloop()


def main():
    board_manipulations()
    pass

if __name__ == '__main__':
    main()
