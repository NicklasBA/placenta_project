import cv2
import glob
from time import sleep

print( "Creating videos from images")
FILES="raw_data/frames/*_6mbar_500fps_*_*"
OUTDIR = "raw_data/videos/"
SLEEPTIME = 5
paths = glob.glob(FILES)
print(f"Outdir is: {OUTDIR}")
print(f"Going to create {len(paths)} videos. Starting in {SLEEPTIME} seconds.")
sleep(SLEEPTIME)

for num, p in enumerate(paths):
    img_array = []
    print(f"Working on {p} - {num+1}/{len(paths)}")
    for filename in glob.glob(p+'/*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    print(f"\tFound and loaded {len(img_array)} images.")
    out = cv2.VideoWriter(f'{OUTDIR}{p}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    print(f"\tWriting to {OUTDIR}{p}.mp4")
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Thank you, next")

print("No more data..?? :(")

