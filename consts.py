import utils 
device = utils.get_device()

# ------------------Model architecture hyperparameters--------------------
L = 12
D = 768
HEADS = 12
PATCH = 16
IMAGE_W = 224
assert IMAGE_W % PATCH == 0, "Image size must be divisible by the patch size"
N = int((IMAGE_W/PATCH)**2)
assert D % HEADS == 0, "The latent vector size D must be divisible by the number of heads"
# To keep num of params constant we set DH = D/HEADS
DH = int(D/HEADS) 
DMSA = HEADS*DH*3
DMLP = 3072
NORM_EPS = 1e-6

# ---------------------Fine-tuning hyperparameters------------------------
LR = 0.003
MOMENTUM = 0.9
EPOCHS = 1
if device.type == "mps":
    BATCH = 32
    MAX_ITERATIONS_PER_EPOCH = 500
else:
    BATCH = 64
    MAX_ITERATIONS_PER_EPOCH = 782
# DROPOUT should be zero if the number of iterations is low
# The paper uses dropout 0.1 for 10000 iterations
DROPOUT = 0.

# ----------------------------Other consts--------------------------------
SEED = 100
MAX_TEST_BATCHES = int(MAX_ITERATIONS_PER_EPOCH*0.15/0.85)
EVALS_PER_EPOCH = 5
IMAGENET_CLASSES_N = 1000
