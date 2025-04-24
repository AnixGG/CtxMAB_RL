import os
import pickle


def check_path(path, make=False):
    if not make:
        return os.path.isfile(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"Ошибка при создании директории {path}: {e}")


def save_pickle(data, path, filename):
    check_path(path, make=True)
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(data, f)


def open_pickle(path):
    file_exist = check_path(path)
    if not file_exist:
        print(f"Файл {path} не существует")
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
