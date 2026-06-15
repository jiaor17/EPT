# An Equivariant Pretrained Transformer for Unified 3D Molecular Representation Learning

Source code for "An Equivariant Pretrained Transformer for Unified 3D Molecular Representation Learning".

### Setup

One can setup the environment via `env.yaml` as

```bash
conda env create -f env.yml
conda activate EPT
```

### Download and Preprocess Data

#### Pretraining Data

Assets for downloading pretraining datasets are listed as follows.

* Small Molecules
  * [GEOM](https://dataverse.harvard.edu/api/access/datafile/4327252)
  * [PCQM4Mv2](http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz)
* Proteins and Complexes
  * PDB(official downloading [script](https://files.wwpdb.org/pub/pdb/software/rsyncPDB.sh))
  * [PDBBind](http://www.pdbbind.org.cn/download.php)

One can preprocess the above raw data into LMDB format via the following scripts.

```bash
# GEOM
python -m scripts.process_data.process_GEOM \
    --base_path <rdkit_dir> --dataset qm9 \
    --out_dir ./processed/GEOM --using_hrdrogen
python -m scripts.process_data.process_GEOM \
    --base_path <rdkit_dir> --dataset drugs \
    --out_dir ./processed/GEOM --using_hrdrogen
# PCQM4Mv2
python -m scripts.process_data.process_GEOM \
    --sdf_file <sdf_dir> \
    --out_dir ./processed/PCQM4M-v2 --using_hrdrogen
# PDB
python -m scripts.process_data.process_PDB_monomer \
	--pdb_dir <pdb_dir> \
	--out_dir ./processed/PDB
# PDBBind
python -m scripts.process_data.process_PDBBind \
	--data_dir <data_dir> \
	--out_dir ./processed/PDBBind
```

#### Downstream Data

- LBA
  One can access the raw data of LBA via this [link](https://zenodo.org/records/4914718) and process the data by

```bash
python -m scripts.process_data.process_LBA.py \
	--base_path <splits_dir>
	--out_dir ./processed/LBA/<id>
```

- MSP
  One can access the raw data of MSP via this [link](https://zenodo.org/records/4962515) and process the data by

```bash
python -m scripts.process_data.process_MSP \
	--base_path <splits_dir>
	--out_dir ./processed/MSP
```
- MPP
  One can acquire the processed QM9 data via the following script. Raw data will be downloaded automatedly.

```bash
python -m scripts.process_data.process_QM9.py \
	--out_dir ./processed/QM9
	--using_hydrogen --download
```

All preprocessed datasets can be acquired from this [link](https://doi.org/10.6084/m9.figshare.c.8197622).

### Pretrain on Multi-Domain Dataset

```python
GPU=0,1,2,3,4,5,6,7 bash execute/pretrain.sh configs/PreTrain/pretrain.yaml
```

The pretrained checkpoint is available at this [google drive](https://drive.google.com/drive/folders/1ISCsnXss6YueYUvAIiR4wpm3k0TGjb44?usp=sharing).

### Finetune on Downstream Tasks

```bash
# LBA
GPU=0 SPLIT=<ID30 or ID60> bash execute/finetune_lba.sh ./configs/LBA <pretrained_ckpt>
# MSP
GPU=0 bash execute/finetune_msp.sh ./configs/MSP <pretrained_ckpt>
# MPP
GPU=0 PROP=<property (e.g. homo)> bash execute/finetune_qm9.sh ./configs/QM9 <pretrained_ckpt>
```

### Evaluation on Downstream Tasks

```
# LBA
python -m evaluation.eval_lba --config ./configs/LBA/<split>/test.yaml --ckpt ./ckpts/LBA/<split> --gpu 0
# MSP
python -m evaluation.eval_msp --config ./configs/MSP --ckpt ./ckpts/MSP --gpu 0
# MPP
python -m evaluation.eval_qm9 --config ./configs/QM9/test.yaml --ckpt ./ckpts/QM9 --gpu 0 dataset.train.property=<prop> dataset.test.property=<prop>
```

## Citation

Please consider citing our work if you find it helpful:

```
@article{jiao2026equivariant,
  title={An equivariant pretrained transformer for unified 3D molecular representation learning},
  author={Jiao, Rui and Kong, Xiangzhe and Zhang, Li and Yu, Ziyang and Ren, Fangyuan and Tan, Wenjuan and Huang, Wenbing and Liu, Yang},
  journal={Nature Communications},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

## Contact

If you have any questions, feel free to reach us at:

Rui Jiao: [jiaor21@mails.tsinghua.edu.cn](mailto:jiaor21@mails.tsinghua.edu.cn)
