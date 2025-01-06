# An Equivalent Pretrained Transformer for Unified 3D Molecular Representation Learning

Source code for "An Equivalent Pretrained Transformer for Unified 3D Molecular Representation Learning".

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
  * PDB
  * PDBBind

One can preprocess the above raw data into LMDB format via the following scripts.

```bash
# GEOM
python -m scripts.process_data.process_GEOM.py \
    --base_path <rdkit_dir> --dataset qm9 \
    --out_dir ./processed/GEOM --using_hrdrogen
python -m scripts.process_data.process_GEOM.py \
    --base_path <rdkit_dir> --dataset drugs \
    --out_dir ./processed/GEOM --using_hrdrogen
# PCQM4Mv2
python -m scripts.process_data.process_GEOM.py \
    --sdf_file <sdf_dir> \
    --out_dir ./processed/PCQM4M-v2 --using_hrdrogen
# PDB
TODO
# PDBBind
TODO

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
  TODO
- MPP
  One can acquire the processed QM9 data via the following script. Raw data will be downloaded automatedly.

```bash
python -m scripts.process_data.process_QM9.py \
	--out_dir ./processed/QM9
	--using_hydrogen --download
```

### Pretrain on Multi-Domain Dataset

```python
GPU=0,1,2,3,4,5,6,7 bash execute/pretrain.sh configs/PreTrain/pretrain.yaml
```

### Finetune on Downstream Tasks

```bash
# LBA
GPU=0 SPLIT=<ID30 or ID60> bash execute/finetune_lba.sh ./configs/LBA <pretrained_ckpt>
# MSP
GPU=0 bash execute/finetune_msp.sh ./configs/MSP <pretrained_ckpt>
# MPP
GPU=0 PROP=<property (e.g. homo)> bash execute/finetune_qm9.sh ./configs/QM9 <pretrained_ckpt>
```
