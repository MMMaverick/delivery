import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='local/nuscenes_scene0103_occ.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    output = Path(config['output'])
    if json.loads((output/'status.json').read_text())['state'] != 'complete':
        raise RuntimeError('Inference incomplete')
    rows = json.loads((output/'results.json').read_text())['frames']
    lines = []
    for index,row in enumerate(rows):
        frame = row['frame']
        cameras = cv2.imread(str(output/f'{frame:06d}_boxes.jpg'))
        bev = cv2.imread(str(output/f'{frame:06d}_occ.jpg'))
        if cameras is None or bev is None:
            raise FileNotFoundError(frame)
        height = cameras.shape[0]
        bev = cv2.resize(bev,(height,height),interpolation=cv2.INTER_NEAREST)
        mosaic = np.concatenate([cameras,bev],axis=1)
        footer = np.full((60,mosaic.shape[1],3),25,dtype=np.uint8)
        cv2.putText(footer,f"Frame {frame}   t={row['timestamp_seconds']:.2f}s | Left: camera detections | Right: semantic OCC top view",
                    (10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
        cv2.putText(footer,'Orange: car | Red: pedestrian | Green: vegetation | Blue-gray: manmade | Gray: road | Black: no retained prediction',
                    (10,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
        path = output/f'{frame:06d}_overview.jpg'
        cv2.imwrite(str(path),np.concatenate([mosaic,footer]))
        seconds = rows[index+1]['timestamp_seconds']-row['timestamp_seconds'] if index+1<len(rows) else row['timestamp_seconds']-rows[index-1]['timestamp_seconds'] if index else .5
        lines.extend([f"file '{path}'",f'duration {seconds:.9f}'])
    lines.append(f"file '{path}'")
    listing = output/'overview_frames.txt'
    listing.write_text('\n'.join(lines)+'\n')
    destination = output/'overview.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(listing),
                    '-vf','fps=10','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],check=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(destination),'-f','null','-'],check=True)
    print(destination)


if __name__ == '__main__':
    main()
