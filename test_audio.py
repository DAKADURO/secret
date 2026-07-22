import imageio_ffmpeg
import subprocess
import os
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
with open('test.ogg', 'wb') as f: f.write(b'OggS' + b'\x00' * 100)
res = subprocess.run([ffmpeg_path, '-i', 'test.ogg', 'test.wav', '-y'], capture_output=True)
print('Return code:', res.returncode)
