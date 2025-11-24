#!/bin/bash
#SBATCH --job-name=rnn_joblib
#SBATCH --output=logs/rnn_out_%j.txt
#SBATCH --error=logs/rnn_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=standard
mkdir -p logs
# --- Módulos/entorno ---
#module purge
#module load python/3.10         # o tu módulo de Python

# (Opcional) activar venv
# source /path/to/venv/bin/activate

cd /home/max.ramirez/pd01/

python3 --version

# --- Evitar oversubscription BLAS ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Joblib usará estos CPUs ---
export N_JOBS="${SLURM_CPUS_PER_TASK}"

# --- Entrenamiento en paralelo con joblib (datos sintéticos por defecto) ---
srun python3 rnn_joblib.py train \
  --n-samples 20000 --n-features 64 --classes 4 \
  --seeds 16 --max-iter 200 \
  --n-jobs "${N_JOBS}" --backend loky --inner-n-jobs 1 \
  --logs-dir ./logs \
  --metrics-json metrics.json \
  --metrics-csv metrics.csv
  #--out-model mlp_model.joblib

# --- (Opcional) Predicción si existe un CSV de prueba ---
#TEST_CSV="data_test.csv"   # pon aquí tu archivo de test (solo features)
#if [ -f "${TEST_CSV}" ]; then
#  srun -c 1 python mlp_joblib_np.py --mode predict \
#    --test-csv "${TEST_CSV}" \
#    --out-model mlp_model.joblib \
#    --out-preds preds.csv
#fi
