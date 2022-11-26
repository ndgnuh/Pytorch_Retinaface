import requests
import sys
import os
from pathlib import Path
from os import path

CACHE_NAME = "retinaface_detector"


def get_file_name_from_url(url: str) -> str:
    return url.split("/")[-1].split("&")[0]


def download(url: str, target: str):
    """
    Download a file from `url` and save to `target`. `target` can be a
    file path or directory. Non-existing target will be created.
    """
    if path.isdir(target):
        target = path.join(target, get_file_name_from_url(url))
    elif target.endswith("/"):
        os.makedirs(target, exist_ok=True)
        target = path.join(target, get_file_name_from_url(url))
    else:
        os.makedirs(path.dirname(target), exist_ok=True)

    response = requests.get(url, allow_redirects=True)

    with open(target, 'wb') as io:
        io.write(response.content)
    return target


def get_cache_dir() -> Path:
    """Locate a platform-appropriate cache directory for flit to use

    Does not ensure that the cache directory exists.
    # This code is from
    # Hopefully it works on windows, lol
    # https://www.programcreek.com/python/?CodeExample=get+cache+dir
    """
    # Linux, Unix, AIX, etc.
    if os.name == 'posix' and sys.platform != 'darwin':
        # use ~/.cache if empty OR not set
        xdg = os.environ.get("XDG_CACHE_HOME", None) \
            or os.path.expanduser('~/.cache')
        return Path(xdg, CACHE_NAME)

    # Mac OS
    elif sys.platform == 'darwin':
        return Path(os.path.expanduser('~'),
                    'Library/Caches/', CACHE_NAME)

    # Windows (hopefully)
    else:
        local = os.environ.get('LOCALAPPDATA', None) \
            or os.path.expanduser('~\\AppData\\Local')
        return Path(local, CACHE_NAME)


def get_model_path(url: str):
    cache_dir = get_cache_dir()
    model_file = get_file_name_from_url(url)
    model_file = path.join(cache_dir, model_file)
    return model_file


def download_model(url: str, force: bool = False):
    model_path = get_model_path(url)
    if path.isfile(model_path) and not force:
        return model_path

    return download(url, model_path)
