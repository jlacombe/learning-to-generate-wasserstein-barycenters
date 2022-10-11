# modified version of the code from
# https://medium.com/@adrianmrit/creating-simple-image-datasets-with-flickr-api-2f19c164d82f
from flickrapi import FlickrAPI
from progress.bar import Bar
import requests
import os
import sys
import time
import os
import time

# replace the values of KEY and SECRET with your own Flickr API key
# KEY = 'your_key'
# SECRET = 'your_secret'

SIZES = ['url_o', 'url_k', 'url_h', 'url_l', 'url_c', 'url_z', 'url_m']  # in order of preference


def get_photos(image_tag):
    extras = ','.join(SIZES)
    flickr = FlickrAPI(KEY, SECRET)
    photos = flickr.walk(text=image_tag,  # it will search by image title and image tags
                            extras=extras,  # get the urls for each size we want
                            privacy_filter=1,  # search only for public photos
                            per_page=50,
                            sort='relevance')  # we want what we are looking for to appear first
    return photos

def get_url(photo):
    for i in range(len(SIZES)):  # makes sure the loop is done in the order we want
        url = photo.get(SIZES[i])
        if url:  # if url is None try with the next size
            return url
        
def get_urls(image_tag, max):
    photos = get_photos(image_tag)
    counter=0
    urls=[]

    for photo in photos:
        if counter < max:
            url = get_url(photo)  # get preffered size url
            if url:
                urls.append(url)
                counter += 1
            # if no url for the desired sizes then try with the next photo
        else:
            break

    return urls

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def download_images(urls, path):
    create_folder(path)  # makes sure path exists

    for img_id, url in enumerate(urls):
        ext = url.split('/')[-1].split('.')[-1]
        image_path = os.path.join(path, '{}.{}'.format(img_id, ext))

        if not os.path.isfile(image_path):  # ignore if already downloaded
            response=requests.get(url,stream=True)

            with open(image_path,'wb') as outfile:
                outfile.write(response.content)

def download(queries, images_per_query):
    for query in queries:
        print('Getting urls for', query)
        urls = get_urls(query, images_per_query)
        
        print('Downloading images for', query)
        path = os.path.join('..', 'datasets', 'flickr', 'imgs', query)

        download_images(urls, path)

def main():
    start_time = time.time()
    images_per_query = 30
    with open('categories.txt', 'r') as f:
        queries = [x.rstrip() for x in f]
    print('Retrieving {} images per query'.format(images_per_query)
        + 'for {} queries => total of {} images.'.format(len(queries), 
                                                         images_per_query*len(queries)))
    download(queries, images_per_query)
    print('Took', round(time.time() - start_time, 2), 'seconds')
    
if __name__ == "__main__":
    main()

