pip install -q jax[tpu] flax optax transformers==4.57.3 datasets diffusers==0.36 torch torchvision Pillow flask -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

DIFF_PATH=$(pip list -v | grep diffusers | awk '{print $3}')
echo "INSTALLED DIFFUSER: $DIFF_PATH"

cp pipeline_flax_stable_diffusion.py "$DIFF_PATH/diffusers/pipelines/stable_diffusion"
mkdir -p $IMAGE_OUTPUT_SAVE_DIR
gsutil cp $GCS_OUTPUT_SAVE_DIR/checklist.txt .

