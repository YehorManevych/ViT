L = 12
D = 768
HEADS = 12
PATCH = 16
IMAGE_W = 224
assert IMAGE_W % PATCH == 0, "Image size must be divisible by the patch size"
N = int((IMAGE_W/PATCH)**2)
assert D % HEADS == 0, "The latent vector size D must be divisible by the number of heads"
DH = int(D/HEADS) # To keep num of params constant we set DH = D/HEADS
DMSA = HEADS*DH*3
DMLP = 3072 # 4 times the D
NORM_EPS = 1e-6
DROPOUT = 0.1

#Fine-tuning
LR = 0.003
MOMENTUM = 0.9

# the paper uses 512 images per batch for fine-tuning
FT_BATCH = 128
#Limiting the number of epochs
EPOCHS = 2

# Limitimg the number of iterations (updates) per epoch
# Consequently it is the number of train batches per epoch
MAX_ITERATIONS_PER_EPOCH = 200