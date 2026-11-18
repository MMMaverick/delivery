import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['others','barrier','bicycle','bus','car','construction_vehicle',
         'motorcycle','pedestrian','traffic_cone','trailer','truck',
         'driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation']
COLORS = np.array([[100,100,100],[50,120,255],[220,50,220],[0,200,255],
                   [255,80,30],[240,180,100],[200,80,180],[255,30,60],
                   [255,100,0],[255,200,100],[255,180,40],[100,100,100],
                   [150,150,150],[140,180,200],[100,130,80],[100,120,180],
                   [40,180,40]], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local/172055_occ_fullview.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    output = Path(config['output'])
    if json.loads((output/'status.json').read_text())['state'] != 'complete':
        raise RuntimeError('Inference incomplete')
    frames = [row['frame'] for row in json.loads((output/'results.json').read_text())['frames']]
    for frame in frames:
        with np.load(output/f'{frame:06d}_occ.npz') as data:
            points = data['point_cloud_range'][:3]+(data['locations']+.5)*float(data['voxel_size'])
            labels = data['labels']
        records = np.empty(len(points), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                               ('red','u1'),('green','u1'),('blue','u1'),('semantic_label','u1')])
        for index, axis in enumerate(['x','y','z']):
            records[axis] = points[:,index]
        for index, channel in enumerate(['red','green','blue']):
            records[channel] = COLORS[labels,index]
        records['semantic_label'] = labels
        header = ('ply\nformat binary_little_endian 1.0\n'
                  'comment Predicted occupied voxel centers, reference ego coordinates in meters\n'
                  f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar semantic_label\nend_header\n')
        destination = output/f'{frame:06d}_occ.ply'
        with destination.open('wb') as stream:
            stream.write(header.encode('ascii'))
            records.tofile(stream)
        if frame == frames[-1]:
            figure = plt.figure(figsize=(12,9))
            axes = figure.add_subplot(111, projection='3d')
            axes.scatter(*points.T, c=COLORS[labels]/255., s=2, depthshade=False)
            axes.set(xlabel='Reference ego X (m)',ylabel='Reference ego Y (m)',zlabel='Z (m)',
                     xlim=(-40,40),ylim=(-40,40),zlim=(-1,5.4),title=f'GeoUP OCC prediction - frame {frame}')
            axes.set_box_aspect((80,80,20))
            axes.view_init(elev=40,azim=-135)
            figure.savefig(output/'pointcloud_preview.png',dpi=160,bbox_inches='tight')
            plt.close(figure)
        print(destination)
    (output/'pointcloud_classes.json').write_text(json.dumps(dict(enumerate(NAMES)),indent=2))


if __name__ == '__main__':
    main()
