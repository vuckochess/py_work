import os
import re


def rename_files():
    base_dir = './Snapshots/'
    dirs = os.listdir(base_dir)

    for subdir in dirs:
        file_names = os.listdir(base_dir + subdir)
        # print(file_names[:5])
        for file in file_names:
            m = re.search(r'_\d{2}[a-z]?\.jpg', file)
            if m:
                full_path = base_dir + subdir + '/'
                old_name = full_path + file
                new_name = full_path + file[:m.start()+1] + '0' + file[m.start()+1:]
                os.rename(old_name, new_name)
                print(old_name, new_name)

def create_json_file():
    import Fen_string_manipulations as fen
    base_dir = './Snapshots/'
    json_file = base_dir + 'board_strings.json'
    dump_file = base_dir + 'snapshots.json'
    pos_dict = fen.fetch_dict_from_json(json_file)

    correlation_dict = {}
    for game in pos_dict:
        pos_dir = base_dir + game + "/"
        files = os.listdir(pos_dir)
        for file, pos in zip(files, pos_dict[game]):
            full_path = pos_dir + file
            correlation_dict[full_path] = pos
    fen.dump_dict_to_json_file(correlation_dict, dump_file)
