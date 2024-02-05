#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from .logger import print_log


def download(url, path, save_file=None, md5=None):
    """
    From https://torchdrug.ai/docs/_modules/torchdrug/utils/file.html#download

    Download a file from the specified url.
    Skip the downloading step if there exists a file satisfying the given MD5.

    Parameters:
        url (str): URL to download
        path (str): path to store the downloaded file
        save_file (str, optional): name of save file. If not specified, infer the file name from the URL.
        md5 (str, optional): MD5 of the file
    """
    from six.moves.urllib.request import urlretrieve

    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file) or compute_md5(save_file) != md5:
        print_log("Downloading %s to %s" % (url, save_file))
        urlretrieve(url, save_file)
    return save_file


def extract(zip_file, member=None):
    """
    From https://torchdrug.ai/docs/_modules/torchdrug/utils/file.html#extract

    Extract files from a zip file. Currently, ``zip``, ``gz``, ``tar.gz``, ``tar`` file types are supported.

    Parameters:
        zip_file (str): file name
        member (str, optional): extract specific member from the zip file.
            If not specified, extract all members.
    """
    import gzip
    import shutil
    import zipfile
    import tarfile

    zip_name, extension = os.path.splitext(zip_file)
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]
    save_path = os.path.dirname(zip_file)

    if extension == ".gz":
        member = os.path.basename(zip_name)
        members = [member]
        save_files = [os.path.join(save_path, member)]
        for _member, save_file in zip(members, save_files):
            with open(zip_file, "rb") as fin:
                fin.seek(-4, 2)
                file_size = struct.unpack("<I", fin.read())[0]
            with gzip.open(zip_file, "rb") as fin:
                if not os.path.exists(save_file) or file_size != os.path.getsize(save_file):
                    print_log("Extracting %s to %s" % (zip_file, save_file))
                    with open(save_file, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
    elif extension in [".tar.gz", ".tgz", ".tar"]:
        tar = tarfile.open(zip_file, "r")
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            print_log("Extracting %s from %s to %s" % (member, zip_file, save_files[0]))
        else:
            members = tar.getnames()
            save_files = [os.path.join(save_path, _member) for _member in members]
            print_log("Extracting %s to %s" % (zip_file, save_path))
        for _member, save_file in zip(members, save_files):
            if tar.getmember(_member).isdir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or tar.getmember(_member).size != os.path.getsize(save_file):
                with tar.extractfile(_member) as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    elif extension == ".zip":
        zipped = zipfile.ZipFile(zip_file)
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            print_log("Extracting %s from %s to %s" % (member, zip_file, save_files[0]))
        else:
            members = zipped.namelist()
            save_files = [os.path.join(save_path, _member) for _member in members]
            print_log("Extracting %s to %s" % (zip_file, save_path))
        for _member, save_file in zip(members, save_files):
            if zipped.getinfo(_member).is_dir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or zipped.getinfo(_member).file_size != os.path.getsize(save_file):
                with zipped.open(_member, "r") as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    else:
        raise ValueError("Unknown file extension `%s`" % extension)

    if len(save_files) == 1:
        return save_files[0]
    else:
        return save_path