# ------------------Model architecture hyperparameters--------------------
L = 12
D = 768
HEADS = 12
PATCH = 16
IMAGE_W = 224
assert IMAGE_W % PATCH == 0, "Image size must be divisible by the patch size"
N = int((IMAGE_W / PATCH) ** 2)
assert D % HEADS == 0, "The latent vector size D must be divisible by the number of heads"
# To keep num of params constant we set DH = D/HEADS
DH = int(D / HEADS)
DMSA = HEADS * DH * 3
DMLP = 3072
NORM_EPS = 1e-6

# ---------------------Fine-tuning hyperparameters------------------------
LR = 0.003
MOMENTUM = 0.9
# BATCH size should be small if compute is limited
# The paper uses batch size of 512 samples
BATCH = 32
# EPOCHS number should be small if compute is limited
# The paper uses 10000 steps to fine-tune the model
# With batch size 512 it's approx. 102 epochs
EPOCHS = 5
# DROPOUT should be zero if the number of iterations is low
# The paper uses dropout 0.1 for 10000 iterations
DROPOUT = 0.0

# ----------------------------Other consts--------------------------------
SEED = 100
IMAGENET_CLASSES_N = 1000
