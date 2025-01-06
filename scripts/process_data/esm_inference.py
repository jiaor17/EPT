#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from tqdm import tqdm

import esm
import torch

from data import MMAPDataset
from data.format import VOCAB
from data.mmap_dataset import create_mmap
from utils.logger import print_log


def process_iterator(dataset: MMAPDataset, model, alphabet, device):
    batch_converter = alphabet.get_batch_converter()
    all_toks = { tok: True for tok in alphabet.all_toks }
    print(all_toks)

    for i, item in enumerate(dataset):
        item_id = dataset._indexes[i][0]
        block_types, segment_ids = item['B'], item['segment_ids']
        seqs, is_prot = [], []
        last_seg_id, cur_seg_is_prot = None, False
        for block_type, seg_id in zip(block_types, segment_ids):
            if seg_id != last_seg_id:
                seqs.append([])
                is_prot.append(cur_seg_is_prot)
                cur_seg_is_prot = False
                last_seg_id = seg_id
            aa = VOCAB.idx_to_symbol(block_type)
            if aa == VOCAB.GLB:
                continue
            elif aa not in all_toks:
                aa = '<mask>'
            else:
                cur_seg_is_prot = True
            seqs[-1].append(aa)
        is_prot.append(cur_seg_is_prot)
        is_prot = is_prot[1:]

        # prepare data
        reprs, is_prot_residue = [], []
        for aas, p in zip(seqs, is_prot):
            data = [ (f'chain', ''.join(aas)) ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations
            with torch.no_grad():
                try:
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]
                    token_representations = token_representations[batch_tokens != alphabet.padding_idx]
                    is_prot_residue.extend(p for _ in aas)
                except torch.cuda.OutOfMemoryError as e:
                    print_log(f'{item_id}: length {len(aas)}, OOM', level='WARN')
                    token_representations = torch.zeros(len(aas), model.embed_dim).float()
                    is_prot_residue.extend(False for _ in aas)
            reprs.extend(token_representations.cpu().tolist())

        yield item_id, (reprs, is_prot_residue), []


def main(args):
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    dataset = MMAPDataset(args.mmap_dir, specify_index=args.index)
    
    # Load ESM-2 model
    if args.esm_ckpt_dir:
        torch.hub.set_dir(args.esm_ckpt_dir)
    # if args.esm_ckpt:
    #     model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.esm_ckpt)
    # else:
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()  # disables dropout for deterministic results

    out_dir = os.path.join(args.mmap_dir, 'esm_embedding') if args.out_dir is None else args.out_dir

    create_mmap(
        process_iterator(dataset, model, alphabet, device),
        out_dir = out_dir,
        total_len = len(dataset)
    )


def parse():
    parser = argparse.ArgumentParser(description='ESM inference')
    parser.add_argument('--mmap_dir', type=str, required=True, help='Binary data')
    parser.add_argument('--index', type=str, default=None, help='Index')
    parser.add_argument('--esm_ckpt_dir', type=str, default=None, help='Directory to save/load the checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--out_dir', type=str, default=None, help='Output Directory, default esm_embedding under mmap_dir')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())