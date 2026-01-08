pip install -q jax[tpu] flax optax transformers==4.57.3 datasets diffusers==0.36 torch torchvision Pillow -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

DIFF_PATH=$(pip list -v | grep diffusers | awk '{print $3}')
echo "INSTALLED DIFFUSER: $DIFF_PATH"

cp pipeline_flax_stable_diffusion.py "$DIFF_PATH/diffusers/pipelines/stable_diffusion"
gsutil cp $GCS_DIR/checklist.txt .
mkdir -p ./sd-finetune
gsutil -m cp -r gs://khang-sd-ft/full ./sd-finetune
mkdir -p $SAVE_DIR
