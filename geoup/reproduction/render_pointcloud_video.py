import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import yaml

from export_occ_cloud import COLORS

ROOT = Path(__file__).resolve().parents[1]


def project(points):
    eye = np.array([85.,-95.,85.])
    forward = -eye/np.linalg.norm(eye)
    right = np.cross(forward,[0.,0.,1.])
    right /= np.linalg.norm(right)
    up = np.cross(right,forward)
    horizontal,vertical = points@right,points@up
    pixels = np.column_stack([640+horizontal*8.,365-vertical*8.])
    return np.rint(pixels).astype(np.int32), points@forward


def render(points, labels, frame, seconds):
    image = np.full((720,1280,3),22,dtype=np.uint8)
    for position in range(-40,41,10):
        for segment in [np.array([[position,-40,-1],[position,40,-1]]),
                        np.array([[-40,position,-1],[40,position,-1]])]:
            pixels,_ = project(segment)
            cv2.line(image,tuple(pixels[0]),tuple(pixels[1]),(55,55,55),1,cv2.LINE_AA)
    pixels,depth = project(points)
    order = np.argsort(depth)
    pixels,labels = pixels[order],labels[order]
    offsets = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    expanded = (pixels[:,None,:]+offsets[None,:,:]).reshape(-1,2)
    colors = np.repeat(COLORS[labels,::-1],len(offsets),axis=0)
    valid = (expanded[:,0]>=0)&(expanded[:,0]<1280)&(expanded[:,1]>=0)&(expanded[:,1]<720)
    expanded,colors = expanded[valid],colors[valid]
    linear = expanded[:,1]*1280+expanded[:,0]
    _,nearest = np.unique(linear,return_index=True)
    image.reshape(-1,3)[linear[nearest]] = colors[nearest]
    for endpoint,color,text in [(np.array([10,0,0]),(40,40,255),'+X'),
                                (np.array([0,10,0]),(40,255,40),'+Y'),
                                (np.array([0,0,5]),(255,180,40),'+Z')]:
        axis,_ = project(np.array([[0,0,0],endpoint]))
        cv2.arrowedLine(image,tuple(axis[0]),tuple(axis[1]),color,2,cv2.LINE_AA,tipLength=.15)
        cv2.putText(image,text,tuple(axis[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    cv2.putText(image,f'GeoUP semantic OCC | frame {frame} | t={seconds:.2f}s | {len(points)} voxels',
                (20,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(image,'Fixed oblique view in current ego frame | 80m x 80m | voxel 0.4m | single-frame prediction',
                (20,58),cv2.FONT_HERSHEY_SIMPLEX,.55,(210,210,210),1,cv2.LINE_AA)
    for index,(label,name) in enumerate([(4,'car'),(7,'pedestrian'),(16,'vegetation'),(15,'manmade'),(11,'road')]):
        left = 20+index*190
        cv2.rectangle(image,(left,679),(left+14,693),tuple(int(value) for value in COLORS[label,::-1]),-1)
        cv2.putText(image,name,(left+22,692),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1,cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='local/172055_occ_fullscene.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    output = Path(config['output'])
    if json.loads((output/'status.json').read_text())['state'] != 'complete':
        raise RuntimeError('Inference incomplete')
    rows = json.loads((output/'results.json').read_text())['frames']
    lines = []
    for index,row in enumerate(rows):
        frame = row['frame']
        with np.load(output/f'{frame:06d}_occ.npz') as data:
            points = data['point_cloud_range'][:3]+(data['locations']+.5)*float(data['voxel_size'])
            labels = data['labels']
        image = render(points,labels,frame,row['timestamp_seconds'])
        path = output/f'{frame:06d}_pointcloud.jpg'
        cv2.imwrite(str(path),image)
        seconds = rows[index+1]['timestamp_seconds']-row['timestamp_seconds'] if index+1<len(rows) else row['timestamp_seconds']-rows[index-1]['timestamp_seconds'] if index else .1
        lines.extend([f"file '{path}'",f'duration {seconds:.9f}'])
        if index%50==0:
            print(f'Rendered {index+1}/{len(rows)} point clouds',flush=True)
    lines.append(f"file '{path}'")
    listing = output/'pointcloud_frames.txt'
    listing.write_text('\n'.join(lines)+'\n')
    destination = output/'pointcloud.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(listing),
                    '-vf','fps=10','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],check=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(destination),'-f','null','-'],check=True)
    print(destination)


if __name__ == '__main__':
    main()
