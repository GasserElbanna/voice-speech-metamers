### srun -t 03:00:00 --mem=15G --pty bash
### ./singularity_jupyter_notebook.sh


unset XDG_RUNTIME_DIR
module add openmind8/anaconda/3-2022.10
source activate metamer310

./jupyter_notebook.sh