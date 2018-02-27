import random
import os

from settings import DATA_FOLDER

if __name__ == '__main__':

    with open(os.path.join(DATA_FOLDER, 'annotations_final.csv')) as annotation_file:
        lines = annotation_file.readlines()
        header = lines[0]
        lines = lines[1:]
        random.shuffle(lines)
        lines = [header] + lines

        with open(os.path.join(DATA_FOLDER, 'annotations_randomized.csv'), 'w') as randomized_file:
            randomized_file.writelines(lines)
            randomized_file.close()