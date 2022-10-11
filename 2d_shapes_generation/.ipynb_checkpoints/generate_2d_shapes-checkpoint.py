import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from multiprocessing import Pool, cpu_count
from scipy.stats import truncnorm
from scipy.ndimage import convolve, gaussian_filter
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.getcwd()))
from io_util import analyse_args


class ShapesGenerator2D:
    '''
    Generator of random 2D shapes. A shape is generated following a CSG fashion:
    we assemble primitives shapes (ellipse, triangle, rectangle, line) together
    using logical operators. Generated shapes can be filled with a random grayscale 
    font (random sinusoidal patterns) or just contours:
    - to create contours: primitives are represented with boolean arrays where True
      means the presence of mass and False its absence. Classical logical operators
      (OR, AND, XOR, NOT) are then applied a certain number of times to create the 
      shape. Finally a Sobel Filter is applied to keep only contours ;
    - to create filled shapes: primitives are represented with real-valued arrays. 
      We fill the shapes with random sinusoidal patterns. We keep a semantic similar 
      to the logical operators OR, AND, XOR and NOT when combining the shapes. 
    '''
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.min_pixels = 50 # minimum number of pixels oomposing a shape
        self.primitives = {
            1: self.gen_random_triangle,
            2: self.gen_random_rectangle,
            3: self.gen_random_ellipse,
            4: self.gen_random_line
        }
    
    def gen_random_polygon_mask(self, npoints):
        polygon = []
        for i in range(npoints):
            x = random.randint(0,self.nrows-1)
            y = random.randint(0,self.ncols-1)
            polygon.append((x,y))
        return polygon
    
    def apply_random_rotation(self, fig):
        rotation_angle = random.randint(0,360)
        fig = fig.rotate(rotation_angle)
        return fig
    
    def gen_random_polygon(self, npoints):
        fig = Image.new('L', (self.nrows,self.ncols), 0)
        polygon = self.gen_random_polygon_mask(npoints)
        ImageDraw.Draw(fig).polygon(polygon, outline=1, fill=1)
        fig = self.apply_random_rotation(fig)
        return np.array(fig, dtype=self.polygon_dtype)
        
    def gen_random_triangle(self):
        return self.gen_random_polygon(3)
    
    def gen_random_rectangle(self):
        fig = Image.new('L', (self.nrows,self.ncols), 0)
        polygon = self.gen_random_polygon_mask(2)
        ImageDraw.Draw(fig).rectangle(polygon, outline=1, fill=1)
        fig = self.apply_random_rotation(fig)
        return np.array(fig, dtype=self.polygon_dtype)
    
    def gen_random_line(self):
        fig = Image.new('L', (self.nrows,self.ncols), 0)
        polygon = self.gen_random_polygon_mask(2)
        ImageDraw.Draw(fig).line(polygon, fill=1, width=random.randint(2,5))
        fig = self.apply_random_rotation(fig)
        return np.array(fig, dtype=self.polygon_dtype)
    
    def gen_random_ellipse(self):
        fig = Image.new('L', (self.nrows,self.ncols), 0)
        x0 = random.randint(0,self.nrows-1)
        y0 = random.randint(0,self.ncols-1)
        x1 = random.randint(x0,self.nrows-1)
        y1 = random.randint(y0,self.ncols-1)
        coords = [(x0, y0), (x1, y1)]
        ImageDraw.Draw(fig).ellipse(coords, outline=1, fill=1)
        fig = self.apply_random_rotation(fig)
        return np.array(fig, dtype=self.polygon_dtype)
    
    def rotate_and_crop(self, img):
        crop_size = 96
        img = Image.fromarray((img * 255 / np.max(img)).astype('uint8'))
        img = self.apply_random_rotation(img)
        img = img.crop((crop_size,crop_size,512-crop_size,512-crop_size))
        img = img.resize((512,512))
        img = np.array(img)
        return img / img.max()

    def gen_random_grayscale_font(self, reg):
        # the random font consists in a weighted sum of sinus with abusive use of random
        x, y = np.meshgrid(range(self.nrows), range(self.ncols), sparse=True)
        w = np.exp(np.random.randn(6))
        w = w / w.sum()
        z =     w[0]*((np.sin(x*random.random()*reg)+1)/2) # +1 and /2 ensures the range is in [0;1]
        z = z + w[1]*((np.sin(y*random.random()*reg)+1)/2)
        z = z + w[2]*((np.sin((x+y)*random.random()*reg)+1)/2)
        z = z + w[3]*((np.sin(((x**2)*random.random())*reg)+1)/2)
        z = z + w[4]*((np.sin(((y**2)*random.random())*reg)+1)/2)
        z = z + w[5]*((np.sin((((x+y)**2)*random.random())*reg)+1)/2)
        z = self.rotate_and_crop(z)
        return z
    
    def gen_random_primitive(self):
        primid = random.randint(1,len(self.primitives))
        primfig = self.primitives[primid]()
        
        if (not self.contours):
            # font_reg is an empirical regularization parameter for the choice of the font
            # values from 1e-1 to 1e-6 provide a set of different fonts
            font_reg = 0.99**random.randint(800,1000)
            
            primfont = self.gen_random_grayscale_font(font_reg)
            for k in range(random.randint(1,5)):
                primfont += self.gen_random_grayscale_font(font_reg)
            primfont /= primfont.sum()

            primfig = primfig * primfont
        
        return primfig
    
    def figs_and(self, fig1, fig2):
        if self.contours:
            fig = fig1 & fig2
        else:
            fig = np.where((fig1 != 0) & (fig2 != 0),
                           (fig1+fig2) / 2, 0)
        return fig
    
    def figs_not(self, fig1, fig2):
        if self.contours:
            fig = fig1 < fig2
        else:
            fig = np.where((fig1 != 0) & (fig2 == 0),
                           fig1, 0)
        return fig
    
    def figs_xor(self, fig1, fig2):
        if self.contours:
            fig = fig1 ^ fig2
        else:
            fig = np.where((fig1 != 0) & (fig2 != 0),
                           0, fig1+fig2)
        return fig
    
    def figs_or(self, fig1, fig2):
        if self.contours:
            fig = fig1 | fig2
        else:
            fig = np.where((fig1 != 0) & (fig2 != 0),
                           (fig1+fig2) / 2, fig1+fig2)
        return fig
            
    def apply_random_boolop(self, old_fig, primitive):
        boolopid = random.randint(1,100)
        
        # highly destructive operations
        # AND: we only keep the positions that both figures share
        if (boolopid >= 1 and boolopid <= 5):   # 5%
            fig = self.figs_and(old_fig, primitive)
        # old_fig NOT primitive: we keep the positions of figure 1 minus the ones shared with figure 2
        elif (boolopid >= 6 and boolopid <= 10): # 5%
            fig = self.figs_not(old_fig, primitive)
            
        # destructive operation
        # primitive NOT old_fig: we keep the positions of figure 2 minus the ones shared with figure 1
        elif (boolopid >= 11 and boolopid <= 20): # 10%
            fig = self.figs_not(primitive, old_fig)
            
        # constructive operations
        # XOR: we only keep the positions that the figure do not share
        elif (boolopid >= 21 and boolopid <= 60): # 40%
            fig = self.figs_xor(old_fig, primitive)
        # OR: we keep all the positions from both figures
        elif (boolopid >= 61 and boolopid <= 100): # 40%
            fig = self.figs_or(old_fig, primitive)
            
        return fig
    
    def get_contours(self, shape, thickness=1., fill_value=255.):
        shape = gaussian_filter(shape, thickness)
        sobelX = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        sobelY = np.array([[-1,-2,-1],
                           [ 0, 0, 0],
                           [ 1, 2, 1]])
        derivX = convolve(shape,sobelX)
        derivY = convolve(shape,sobelY)
        gradient = derivX + derivY * 1j
        contours = np.absolute(gradient)
        contours[contours > 0.] = fill_value
        return contours
    
    def gen_random_fig(self, depth, contours):
        self.contours = contours
        self.polygon_dtype = 'bool' if self.contours else 'float'
        fig = self.gen_random_primitive()
        while np.count_nonzero(fig) < self.min_pixels:
            fig = self.gen_random_primitive()
            
        for i in range(depth):
            current_primitive = self.gen_random_primitive()
            tmp_fig = self.apply_random_boolop(fig, current_primitive)
            # restarts if the previous operation was too destructive
            while np.count_nonzero(tmp_fig) < self.min_pixels:
                current_primitive = self.gen_random_primitive()
                tmp_fig = self.apply_random_boolop(fig, current_primitive)
            fig = tmp_fig
        
        if (self.contours):
            fig = self.get_contours(fig, thickness=0.25)
            if np.count_nonzero(fig) < self.min_pixels:
                fig = self.gen_random_fig(depth, contours)
        
        # each image is normalized to have its min value = 0 and its max value = 255
        fig = (fig / fig.max()) * 255.
        fig = fig.astype(np.uint8)
        return fig

def truncated_normal(size=1, mean=0, std=1, low=0, upp=100):
    distrib = truncnorm((low - mean) / std, 
                        (upp - mean) / std, 
                        loc=mean, scale=std)
    return distrib.rvs(size=size)


def main():
    t1 = time.time()
    args = analyse_args([
        ['i',      'img_size', lambda x: int(x), 512],
        ['n',      'n_images', lambda x: int(x), 1000],
        ['p',         'nproc', lambda x: int(x), 10],
        ['b',    'batch_size', lambda x: int(x), 100],
        ['c',    'chunk_size', lambda x: int(x), 1],
        ['d',  'dataset_path', lambda x: str(x), os.path.join('.', 'input_shapes_test.h5')],
        ['w', 'with_contours', lambda x: x == 'True', True]
    ])
    
    dsname = args['dataset_path'].split('/')[-1].split('.')[0]
    sgen = ShapesGenerator2D(args['img_size'], args['img_size'])

    print('Preparing depth distribution...')
    min_depth = 0
    max_depth = 50
    distribs = [
        {'type': np.random.uniform, 'params': {'low': min_depth, 'high': max_depth+1}},
        {'type':  truncated_normal, 'params': {'mean': min_depth, 'std': 2.5, 
                                               'low': min_depth, 'upp': max_depth+1}},
        {'type':  truncated_normal, 'params': {'mean': max_depth+1, 'std': 2.5, 
                                               'low': min_depth, 'upp': max_depth+1}},
    ]
    n_distribs = len(distribs)
    weights = [1/3,1/3,1/3]
    with_contours = np.array([args['with_contours'] for _ in range(args['batch_size'])])

    data = np.zeros((args['n_images'], n_distribs))
    for idx, distr in enumerate(distribs):
        data[:, idx] = distr['type'](size=(args['n_images'],), **distr['params'])
    random_idx = np.random.choice(np.arange(n_distribs), size=(args['n_images'],), p=weights)
    depths = data[np.arange(args['n_images']), random_idx]
    depths = np.floor(depths).astype(int)
    plt.figure()
    plt.hist(depths, bins=max_depth+1)
    plt.savefig('{}_depths_distribution.png'.format(dsname))
    print('Done.')
    
    print('Creating h5 dataset...')
    with h5py.File(args['dataset_path'], 'w') as f:
        dset = f.create_dataset('input_shapes', (args['n_images'],args['img_size'],args['img_size']), 
                                compression='gzip', shuffle=True, dtype=np.uint8, 
                                chunks=(args['chunk_size'],args['img_size'],args['img_size']))
        for i in range(0, args['n_images'], args['batch_size']):
            print('{}-{}'.format(i,i+args['batch_size']))
            figs = []
            with Pool(args['nproc']) as p:
                figs = p.starmap(sgen.gen_random_fig, zip(depths[i:i+args['batch_size']], with_contours))
            figs = np.stack(figs)
            dset[i:i+args['batch_size']] = figs
            del figs
    
    print('Done.')
    deltat = time.time() - t1
    print('Elapsed Time: {:.2f}s for {} images'.format(deltat, args['n_images']))
    with open('{}_generation_time.txt'.format(dsname), 'w') as f:
        f.write('Elapsed Time: {:.2f}s for {} images'.format(deltat, args['n_images']))
    
if __name__ == "__main__":
    main()
