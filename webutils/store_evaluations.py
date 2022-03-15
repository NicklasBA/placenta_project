import os
import toml

stored_path = "webutils/data/stored_results.toml"


def _write_results_to_disk(result, path):
    try:
        os.makedirs(path.rsplit(os.sep)[0])
    except FileExistsError:
        pass
    with open(path, "w") as toml_file:
        toml.dump(result, toml_file)


def _strip_filename(name):
    return os.path.basename(name.strip())


# Load results from disk or write a new file
try:
    stored_result = toml.load(stored_path)
except FileNotFoundError:
    _write_results_to_disk({"CLASSIFIER": {"D": 0, "NS": 1}, "RESULTS": {}}, stored_path)


def store_result(name, classifier, no_d, no_ns):
    name = _strip_filename(name)
    stored_result["RESULTS"][name]["classifier"] = classifier
    stored_result["RESULTS"][name]["no_ns"] = no_ns
    stored_result["RESULTS"][name]["no_d"] = no_d
    toml.dump(stored_result, stored_path)


def get_result(name, full=False):
    name = _strip_filename(name)
    if full:
        return stored_result
    try:
        return stored_result["RESULTS"][name]
    except KeyError:
        return None


def get_user_downloads_folder():
    folders_to_test = [os.path.join(os.path.expanduser("~"), "Downloads"),
                       os.path.join(os.path.expanduser("~"), "downloads"),
                       os.path.join(os.path.expanduser("~"), "Download"),
                       os.path.join(os.path.expanduser("~"), "download")]
    folder = None
    for f in folders_to_test:
        if os.path.isdir(f):
            folder = os.path.join(f, "cullunator_videos")
            try:
                os.mkdir(folder)
            except FileExistsError:
                pass
            return folder

    if folder is None:
        folder = "webpage-downloads"
        os.mkdir(folder)
        return folder

    raise RuntimeError("Download folder could not be found")
