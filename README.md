# Sparse Additive Mechanism Shift VAE (SAMS-VAE)

Code accompanying "Modeling Cellular Perturbations with Sparse Additive Mechanism Shift Variational Autoencoder" (Bereket & Karaletsos, NeurIPS 2023)

### Install Environment

Linux
```
conda create --name sams_vae --file env/conda-linux-64.lock
conda activate sams_vae
pip install -e .
```

Mac
```
conda create --name sams_vae --file env/conda-osx-arm64.lock
conda activate sams_vae
pip install -e .
```

The results in the paper were generated using the Linux environment.

### Download datasets

The perturbseq datasets analyzed in our paper can be downloaded by running:
```commandline
python download_datasets.py [--replogle] [--norman]
```
The Replogle dataset is approximately 550MB, and the Norman dataset is approximately 1.6GB. Each dataset will be saved to the directory `datasets/`

To reuse these cached files while running experiments, set the environment variable `SAMS_VAE_DATASET_DIR` to the absolute path of `datasets/`

To avoid having to repeatedly set the variable, the following script can be used to set the variable when activating the `sams_vae` environment. Make sure to replace the path on your machine in the script:
```commandline
conda activate sams_vae

cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d

echo "#\!/bin/sh" > ./etc/conda/activate.d/env_vars.sh


### replace {/sams_vae_path} with the absolute path to this repository
echo "export SAMS_VAE_DATASET_DIR={/sams_vae_path}/datasets/" >> ./etc/conda/activate.d/env_vars.sh

echo "#\!/bin/sh" > ./etc/conda/deactivate.d/env_vars.sh
echo "unset SAMS_VAE_DATASET_DIR" >> ./etc/conda/deactivate.d/env_vars.sh

# Need to reactivate the environment to see the changes
conda activate sams_vae
```

## Training models

The easiest way to train a model is specify a config file (eg `tests/models/sams_vae_correlated.yaml`) with data, model, and training hyperparameters
(including whether to record results locally or remotely on Weights and Biases). To train using a specified config, run

```python
python train.py --config [path/to/config.yaml]`
```

For larger experiments, we provide support for wandb sweeps using redun. To launch a training sweep, run
```commandline
redun run launch_sweep.py launch_sweep --config-path [path/to/sweep_config/yaml] --num-agents [max-agents]
```
redun can be used to run jobs in parallel on a compute cluster. To do so, add a redun executor in `.redun/redun.ini` and update the executors in `launch_sweep.py` (see https://insitro.github.io/redun/executors.html for more info on defining an executor).
By default, training jobs are run locally.


## Replicating results

We provide sweep configurations, python scripts, and jupyter notebooks to replicate each analysis from the paper in the `paper/experiments/` directory.
Additionally, we provide our precomputed metrics and checkpoints for download to allow exploration of the results without rerunning all experiments.
Detailed instructions for replicating each analysis are available in the README files of the `paper/experiments/` directory.
