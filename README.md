# Dataset 
@ auther Yifan Chen

## channelB (*-B.tif)

In tiff format, 3 channel RGB format (W1920 × H1440 RGB). (I have manually adjusted the Brightness/Contrast)

To read into single channel grayscale image in python, please use 
```{r}
np.array(Image.open(IMAGE_PATH).convert('L'))
``` 

The blue channel images are the DNA after fluorescence staining. So it marks the region of the nuclear. 

## channelG (*-G.tif)

In tiff format, 3 channel RGB format (W1920 × H1440 RGB). (I have manually adjusted the Brightness/Contrast)

To read into single channel grayscale image in python, please use

```{r}
np.array(Image.open(IMAGE_PATH).convert('L'))
``` 

The green channel images are the tubulin structure after fluorescence staining. We need to use the tubulin (in green channel) to identify our cell-of-interest. To identify cells during mitosis, only using the green tubulin (there is a structure we call it midbody) can we tell that the daughter cells came from the same mother.


## 2channel_image (*.npy)

Stacking the B and G channel images together, as a numpy array in .npy format (1440 x 1920 x 2 array).

It is formed by 
```{r} 
2channelImg = np.stack((imgB,imgG),axis=-1)
```


## 3channel_image (*-RGB.png)

Merge the Blue and Green channels together as a single RGB image using FIJI-ImageJ (a software, download here https://imagej.net/Fiji).


## label (*-label.png)

Are binary images with 1 indicating the nuclei in mitosis and 0 indicating background.

**Preprocess from ROI** The original labels are in the form of ROI.zip, which are annotations done by lab students and can be open using Fiji. To generate a binary label image, you need to create a boundary annotated image, and use python script to fill the holes by 
```{r}
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

def assign_nn(arr): 
# input array:  postive elements on boundary 
    annotation = np.ones(arr.shape, dtype='uint8')
    background = np.where((255-arr)>arr)
    annotation[background] = 0
    return binary_fill_holes(annotation).astype(np.uint8)
```


