param_pkl=$(srun '--exclude=cn[65-69,71-136,153-256,265-320,325-328]' run_container.sh python run_nb_with_param_0.py)

for x in ${param_pkl}; do
    sbatch run_container.sh python run_nb_with_param_1.py classification_slex_slim.ipynb ${x};
    done