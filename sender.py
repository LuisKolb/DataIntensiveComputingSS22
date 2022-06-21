from ast import arg
import sys
import argparse
from os import listdir, walk
from os.path import isfile, join
import concurrent.futures
from urllib import response
import numpy as np
import subprocess


p = argparse.ArgumentParser(
    description='run object detection on a bunch of images, distributed on a cluster')
p.add_argument('--output', action='store_true', default=False,
               help='save the annotated images? (default: don\'t save)')
p.add_argument('--nodes', type=str, nargs='+',
               help='endpoint(s) to use (format: --nodes url:port [url:port ...]')
p.add_argument(
    'img_folder', help='path to the folder of the images to run detection on')


def send_curls(node, path_list, base_dir, output):
    res = []
    for path in path_list:
        if path:

            if output:
                o = '&output=1'
            else:
                o = ''

            # endpoint will be called like curl http://url:port/api/detect -d "input=/images/filename.jpg&output=1"

            # a few testing "commands"
            res.append(subprocess.run(
                ['echo', 'curl', 'http://'+node+'/api/detect', '-d', '"input='+base_dir+path+o+'"']))
            #res.append(subprocess.run(['sleep', '5']))

            # the actual command once the endpoints are done
            #res = subprocess.run(['curl', 'http://'+node+'/api/detect', '-d', '"input='+path+o+'"'])

    return res


def process_result(future):
    # print(future.result())
    pass


def main(images_folder, output, nodes, **kwargs):
    # scan folder for images
    _, _, img_paths = next(walk(images_folder))
    if not img_paths:
        return 'folder empty'

    chunked_paths = np.array_split(img_paths, len(nodes))

    # distribute load
    #

    executor = concurrent.futures.ProcessPoolExecutor()
    futures = []
    for i in range(0, len(nodes)):
        futures.append(executor.submit(
            send_curls, nodes[i], chunked_paths[i], images_folder,  output))

    for future in futures:
        future.add_done_callback(process_result)
    executor.shutdown()

    return 'success'


#
# call the script like (image dir first!)
# python sender.py /folder/ --output --nodes 127.0.0.1:5000
#
if __name__ == '__main__':
    args = p.parse_args()
    main(args.img_folder, args.output, args.nodes)
