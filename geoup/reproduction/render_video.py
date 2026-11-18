import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local/172055_validated.yaml')
    arguments = parser.parse_args()
    config = yaml.safe_load((ROOT/arguments.config).read_text())
    output = Path(config['output'])
    status = json.loads((output/'status.json').read_text())
    if status['state'] != 'complete':
        raise RuntimeError('Inference has not passed')
    rows = json.loads((output/'results.json').read_text())['frames']
    times = [row.get('timestamp_seconds',index*.2) for index,row in enumerate(rows)]
    for label in ['boxes','depth'] + (['occ'] if 'occ' in config['tasks'] else []):
        listing = output/f'{label}_frames.txt'
        lines = []
        for index, frame in enumerate([row['frame'] for row in rows]):
            image = output/f'{frame:06d}_{label}.jpg'
            if not image.is_file():
                raise FileNotFoundError(image)
            duration = times[index+1]-times[index] if index+1<len(times) else (times[-1]-times[-2] if len(times)>1 else .2)
            lines.extend([f"file '{image}'",f'duration {duration:.9f}'])
        lines.append(f"file '{image}'")
        listing.write_text('\n'.join(lines)+'\n')
        destination = output/f'{label}.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(listing),
            '-vf','fps=10','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],check=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(destination),'-f','null','-'],check=True)
        print(destination)


if __name__ == '__main__':
    main()
