import glob
from time import sleep
from tqdm import tqdm
import subprocess
import argparse
import os

SLEEPTIME = 5
FFMPEG_PATH = subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if FFMPEG_PATH.returncode == 0:
    FFMPEG_PATH = FFMPEG_PATH.stdout.decode('utf-8').strip()
    print(f"Using ffmpeg from {FFMPEG_PATH}")
else:
    raise RuntimeError("Could not locate ffmpeg path, please make sure it is installed. Run 'conda install -c ")


def runFFmpeg(input_folder, output_file, overwrite=False):

    if overwrite == True:
        ov_flag = "-y"
    else:
        ov_flag = "-n"

    commands_list = [
        FFMPEG_PATH,
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(input_folder, "*.png"),
        "-crf",
        "0",
        "-r",
        "15",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-hide_banner",
        "-loglevel",
        "error",
        ov_flag,
        output_file
    ]

    run = subprocess.run(commands_list)
    return run


def main(filename_pattern, output_dir, overwrite):
    paths = glob.glob(filename_pattern)
    err_paths = []
    pbar = tqdm(range(len(paths)), desc="Converting folders")
    for num in pbar:
        p = paths[num]
        p_pattern = os.path.join(p, "*.png")
        imagepathsw = sorted(glob.glob(p_pattern))
        pbar.set_postfix_str(s=f'{p}, {len(imagepathsw)} images', refresh=True)

        #filename_file_path = os.path.join(p, "sorted_filenames.txt")
        #textfile = open(os.path.join(filename_file_path, "sorted_filenames.txt"), "w")
        #for element in imagepathsw:
        #    textfile.write(element + "\n")
        #textfile.close()

        run = runFFmpeg(p, os.path.join(output_dir, os.path.basename(p)+".mp4"), overwrite)
        if run.returncode == 0:
            pass  # Successfull
        else:
            print(f"There was an error running your FFmpeg for path {p}")
            err_paths.append(p)

    print(f"No more data..?? :(, thats it. {len(paths)} folders processed")
    if err_paths:
        print(f"{len(err_paths)} folders resulted in an error")
        for err in err_paths:
            print(err)
        return 1
    else:
        return 0


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Create videos from folder of image files.")
    parser.add_argument("-i", "--input_folder_pattern",
                        default="raw_data/frames/*_6mbar_500fps_*_*",
                        type=str,
                        help="Patteren of folders containing images, e.g. (the default) "
                             "'raw_data/frames/*_6mbar_500fps_*_*'",
                        )
    parser.add_argument("-o", "--output",
                        default="raw_data/videos/",
                        type=str,
                        help="Output folder, e.g. (the default) "
                             "'raw_data/videos/'",
                        )
    parser.add_argument("-ow", "--overwrite",
                        default=False,
                        action="store_true",
                        help="Should existing files be overwritten. Default: False",
                        )
    args = parser.parse_args()

    files = args.input_folder_pattern
    outdir = args.output


    numpaths = len(glob.glob(files))
    print(f"Outdir is: {outdir}")
    print(f"Going to create {numpaths} videos. Starting in {SLEEPTIME} seconds.")
    print(f"Overwriting exiting files: {args.overwrite}")
    sleep(SLEEPTIME)

    main(files, outdir, args.overwrite)
