#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import shutil
import argparse
from collections import defaultdict

import numpy as np



def exec_mmseq(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def clustering(fasta, tmp_dir, seq_id):

    # clustering
    db = os.path.join(tmp_dir, 'DB')
    cmd = f'mmseqs createdb {fasta} {db}'
    exec_mmseq(cmd)
    db_clustered = os.path.join(tmp_dir, 'DB_clu')
    cmd = f'mmseqs cluster {db} {db_clustered} {tmp_dir} --min-seq-id {seq_id} -c 0.95 --cov-mode 1'  # simlarity > 0.4 in the same cluster
    res = exec_mmseq(cmd)
    num_clusters = re.findall(r'Number of clusters: (\d+)', res)
    if not len(num_clusters):
        raise ValueError('cluster failed!')

    # write clustering results
    tsv = os.path.join(tmp_dir, 'DB_clu.tsv')
    cmd = f'mmseqs createtsv {db} {db} {db_clustered} {tsv}'
    exec_mmseq(cmd)
    
    # read tsv of class \t pdb
    with open(tsv, 'r') as fin:
        entries = fin.read().strip().split('\n')
    id2clu, clu2id = {}, defaultdict(list)
    for entry in entries:
        cluster, _id = entry.strip().split('\t')
        id2clu[_id] = cluster

    for _id in id2clu:
        cluster = id2clu[_id]
        clu2id[cluster].append(_id)
    
    clu_cnt = [len(clu2id[clu]) for clu in clu2id]
    print(f'cluster number: {len(clu2id)}, member number ' +
          f'mean: {np.mean(clu_cnt)}, min: {min(clu_cnt)}, ' +
          f'max: {max(clu_cnt)}')
    
    return id2clu, clu2id


def get_non_redundant(mmap_dir):
    np.random.seed(12)
    index_path = os.path.join(mmap_dir, 'index.txt')
    parent_dir = mmap_dir

    # load index file
    items = {}
    with open(index_path, 'r') as fin:
        lines = fin.readlines()
        for line_id, line in enumerate(lines):
            values = line.strip().split('\t')
            _id, seq = values[0], values[-1]
            items[_id] = (seq, line)

    # make temporary directory
    tmp_dir = os.path.join(parent_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        raise ValueError(f'Working directory {tmp_dir} exists!')
    
    # 1. get non-redundant dimer by 90% seq-id
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    # receptor
    with open(fasta, 'w') as fout:
        for _id in items:
            fout.write(f'>{_id}\n{items[_id][0]}\n')
    id2clu, clu2id = clustering(fasta, tmp_dir, 0.9)
    shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    # overall clustering
    non_redundant = []
    for clu in clu2id:
        ids = clu2id[clu]
        non_redundant.append(np.random.choice(ids))
    print(f'Non-redundant entries: {len(non_redundant)}')

    # 2. construct non_redundant items
    items = { _id: items[_id] for _id in non_redundant }
    return items


def main(args):
    items = get_non_redundant(args.mmap_dir)
    out = open(os.path.join(args.mmap_dir, 'non_redundant_index.txt'), 'w')
    for _id in items:
        out.write(items[_id][-1])
    out.close()


def parse():
    parser = argparse.ArgumentParser(description='non redundant pdb index')
    parser.add_argument('--mmap_dir', type=str, required=True, help='Directory of the mmap')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())