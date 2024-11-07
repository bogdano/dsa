import numpy as np
from matplotlib import pyplot as plt
import PIL.Image as Image
import segmentationClass

# INSTANTIATE SEGMENTATION OBJECT
obj = segmentationClass.segmentationClass()

# INPUT --> LOAD IMAGE
##########################################################################
# file_name = 'test-10x10.png'                
# file_name = 'test-15x15.png'                
file_name = 'test-30x30.png'                


# INPUT --> SET SEGMENTATION PARAMETERS
##########################################################################
# obj.x_a = np.array([3,4]);                  # Foreground pixel coordinate
# obj.x_a = np.array([2,2]);                  # 15x15 image
obj.x_a = np.array([14,18]);                # 30x30 image

obj.x_b = np.array([0,0]);                  # Background pixel coordinate
obj.p0 = 1;                                 # p0 parameter

# SEGMENT THE IMAGE
I = Image.open(file_name).convert('RGB')    # convert to RGB, just in case
I = np.array(I)                             # convert to NxNx3 numpy array
t = obj.segmentImage(I);

# PLOT RESULTS
fig, axs = plt.subplots(1,2, figsize=(10, 5), dpi=100)
fig.suptitle('Input and segmentation')
axs[0].imshow(I.astype(np.uint8), interpolation='nearest')
axs[0].set_title(f"Input image ({I.shape[0]}x{I.shape[1]})")
axs[1].imshow(255*t.astype(np.uint8), cmap='gray')
axs[1].set_title("Binary segmentation")
plt.show()

# PRINT ADJACENCY MATRIX (0,0) and (1,0)
A = obj.constructAdjacencyMatrix(I)
print(A[[0,3],:])
