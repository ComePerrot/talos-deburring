# pfcalcul LAAS

This file explain how to run the training on the pf-calcul at LAAS.
In order to use it, access needs to be granted by sysadmin,
More information about the pf can be found on the [website](https://pfcalcul.laas.fr/).

## Building apptainer image

The training runs inside an apptainer container to avoid having to install all
the dependencies directly on the cluster.

To build the apptainer image simply run:

```bash
apptainer build rl.sif apptainer/rl.def
```

Or

```bash
apptainer build --sandbox rl.sifdir apptainer/rl_sandbox.def
```

Building a sandbox image allows to change the content of the image without rebuilding everything
which is more convenient during prototyping phases.

## Setting up environment

The working directory of the cluster also needs to be set up.
You can use the directory `/pfcalcul/work/$USER` which can be accessed by the working nodes.

This directory is not saved, so any important data should be copied somwhere else to be archived.

- Set up the working directory on the cluster

```bash
cd /pfcalcul/work/$USER
mkdir logs
mkdir output
```

- Copy necessary files into the working directory of the cluster:
  - config folder
  - `schedule.sh` file
  - apptainer image (`rl.sif` or `rl.sifdir` if you followed instruction given in the last part)

## Running training

Once everything has been set up on the pf-calcul,
the training can be launched with the following command:

```bash
sbatch ./schedule.sh
```

This command will automatically bind the `logs` folder and the `config` folder to apptainer.
This allow the user to change the configuration of the training without having to access directly the image.

## Tensorboard Integration

Tensorboard integration requires a virtual env containing tensorboard:

```bash
cd /pfcalcul/work/$USER
python3 -m venv tensorboard
source tensorboard/bin/activate
pip install -U pip
pip install tensorboard
```

To use tensorboard on pfcalul:

- Port 6006 of the server need to be forwarded as follows:

```bash
ssh -L 16006:127.0.0.1:6006 $USER@pfcalcul.laas.fr
```

- Then tensorboard can be run directly on the server:

```bash
cd /pfcalcul/work/$USER
source tensorboard/bin/activate # Activate virtual env containing tensorboard
tensorboard --bind_all --logdir logs
```

- The tensorboard can be viewed on the local machine at:

```bash
http://127.0.0.1:16006
```
