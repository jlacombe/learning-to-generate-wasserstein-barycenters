import sys
import os
import numpy as np
from skimage.transform import resize
from imageio import imread, imwrite

# allows imports from parent directories
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
from io_util import analyse_args

def main():
    args = analyse_args([
        ['f', 'img_folder', lambda x: str(x), 'input_imgs'],
        ['r', 'res_folder', lambda x: str(x), 'input_imgs_28x28'],
        ['i',     'img_id', lambda x: str(x), 'cat1'],
        ['s',       'size', lambda x: [int(n) for n in x.split(',')], [28,28]]
    ])

    input_img_path = os.path.join(args['img_folder'], args['img_id'])
    res_img_path = os.path.join(args['res_folder'], args['img_id'])
    img = imread(input_img_path + '.png')
    img = resize(img, args['size'])
    img = ((img / img.max()) * 255.).astype(np.uint8)
    imwrite(res_img_path + '.png', img)
    
if __name__ == "__main__":
    main()
