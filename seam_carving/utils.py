import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def Sobel_operator(image):
    # 1. Define Sobel kernels (already flipped)
    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)
    
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)
    
    # 2. Compute the gradient in both x and y directions
    h, w = image.shape
    dx = np.zeros_like(image, dtype=np.float32)
    dy = np.zeros_like(image, dtype=np.float32)
    
    padded = np.pad(image, ((1, 1), (1, 1)), 'constant')
    
    # Apply Sobel operators manually
    # I skip the border pixels (1 pixel on each side) for simplicity
    for i in range(1, h+1):
        for j in range(1, w+1):
            # Extract 3x3 neighborhood
            region = padded[i-1:i+2, j-1:j+2]
            
            # Compute x and y gradients
            dx[i-1, j-1] = np.sum(region * sobel_x)
            dy[i-1, j-1] = np.sum(region * sobel_y)

    return dx, dy

def compute_energy(image):
    """
    Compute energy map of an image using gradient magnitude.
    :param image: Input image as numpy array (H, W, 3)
    :return: Energy map (H, W)
    """
    # Convert RGB-style to grayscale first
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    elif len(image.shape) == 2:
        gray = image
    else:
        raise ValueError("Image must be 2-dim or 3-dim")
    
    # Compute gradients using Sobel operator
    dx, dy = Sobel_operator(gray)
    
    # Energy is the magnitude of the gradient
    energy = np.abs(dx) + np.abs(dy)
    
    return energy

def find_vertical_seam(energy, num_seam=1):
    """
    Find vertical seam with minimum energy using dynamic programming.
    :param energy: Energy map (H, W)
    :return: List of column indices for the seam (length H)
    """
    h, w = energy.shape
    cost = energy.copy()
    # Record the column indices of minimal cost paths
    path = np.zeros_like(cost, dtype=np.int32)
    mask = np.zeros((h, w), dtype=bool)
    
    seams = []
    
    # Dynamic programming to compute minimal cost paths
    cycle = False
    for d in range(num_seam):
        if cycle:
            print("cycle~")
            break
        refresh = True
        while refresh:
            refresh = False
            print(f"round {d}")
            for i in range(1, h):
                for j in range(w):
                    if j == 0:      # left boundary: can only go right or up
                        min_idx = np.argmin(cost[i-1, j:j+2])
                        cost[i, j] += cost[i-1, j + min_idx]
                        path[i, j] = j + min_idx
                    elif j == w-1:  # right boundary: can only go left or up
                        min_idx = np.argmin(cost[i-1, j-1:j+1])
                        cost[i, j] += cost[i-1, j-1 + min_idx]
                        path[i, j] = j-1 + min_idx
                    else:
                        min_idx = np.argmin(cost[i-1, j-1:j+2])
                        cost[i, j] += cost[i-1, j-1 + min_idx]
                        path[i, j] = j-1 + min_idx

            seam = [np.argmin(cost[-1])]
            if cost[-1, seam] == np.inf:
                cycle = True
                break
            final_change = h-1
            for i in range(h-1, 0, -1):
                next = path[i, seam[-1]]
                if cost[i-1, next] == np.inf:
                    refresh = True
                    final_change = i
                    print("refresh ===================================")
                    break
                seam.append(next)
                
            seam_reverse = seam[::-1]
            if not refresh:
                final_change = 0
                seams.append(seam_reverse)
            
            # mask the seam
            for i in range(h - final_change):
                mask[i+final_change, seam_reverse[i]] = True
            
            # update for next round
            energy[mask] = np.inf
            cost = energy.copy()
            path = np.zeros_like(cost, dtype=np.int32)
    
    if cycle:
        length = len(seams)
        for i in range(length, num_seam):
            seams.append(seams[i % length])
        print(f"final length: {len(seams)}")
        
    return seams

def find_horizontal_seam(energy, num_seam=1):
    """
    Find horizontal seam with minimum energy by transposing the image.
    :param energy: Energy map (H, W)
    :return: List of row indices for the seam (length W)
    """
    return find_vertical_seam(energy.T, num_seam)


def remove_vertical_seam(image, seam):
    """
    Remove a vertical seam from the image.
    :param image: Input image (H, W, 3) or (H, W)
    :param seam: List of column indices (length H)
    :return: Image with seam removed (H, W-1, 3) or (H, W-1)
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        mask = np.ones((h, w), dtype=bool)
        try:
            for i in range(h):
                mask[i, seam[i]] = False
        except IndexError:
            print("Error: Seam length does not match image height")
            import pdb; pdb.set_trace()
        return image[mask].reshape((h, w-1, c))
    else:
        h, w = image.shape
        mask = np.ones((h, w), dtype=bool)
        for i in range(h):
            mask[i, seam[i]] = False
        return image[mask].reshape((h, w-1))


def remove_horizontal_seam(image, seam):
    """
    Remove a horizontal seam from the image.
    :param image: Input image (H, W, 3) or (H, W)
    :param seam: List of row indices (length W)
    :return: Image with seam removed (H-1, W, 3) or (H-1, W)
    """
    if len(image.shape) == 3:
        return remove_vertical_seam(np.transpose(image, (1, 0, 2)), seam).transpose(1, 0, 2)
    else:
        return remove_vertical_seam(image.T, seam).T


def add_vertical_seam(image, seams):
    """
    Add a vertical seam to the image by duplicating pixels and averaging neighbors.
    :param image: Input image (H, W, 3) or (H, W)
    :param seam: List of column indices (length H)
    :return: Image with seam added (H, W+1, 3) or (H, W+1)
    """
    if seams == [] or seams is None:
        return image
    
    h, w = image.shape[:2]
    
    seams = np.array(seams)
    seams_T = seams.T
    seams_T = [sorted(row) for row in seams_T]
    seams = np.array(seams_T).T
    
    count = 0
    if len(image.shape) == 3:
        for seam in seams:
            output = np.zeros((h, w+1, 3))
            for i in range(h):
                j = seam[i] + count
                if j == 0:
                    # Left boundary case
                    output[i, j] = image[i, j]
                    output[i, j+1] = (image[i, j] / 2 + image[i, j+1] / 2).astype(int)
                    output[i, j+2:] = image[i, j+1:]
                else:
                    output[i, :j] = image[i, :j]
                    output[i, j] = (image[i, j-1] / 2 + image[i, j] / 2).astype(int)
                    output[i, j+1:] = image[i, j:]
            image = output
            w += 1
            count += 1

    else:
        for seam in seams:
            output = np.zeros((h, w+1))
            for i in range(h):
                j = seam[i] + count
                if j == 0:
                    output[i, j] = image[i, j]
                    output[i, j+1] = (image[i, j] / 2 + image[i, j+1] / 2).astype(int) 
                    output[i, j+2:] = image[i, j+1:]
                else:
                    output[i, :j] = image[i, :j]
                    output[i, j] = (image[i, j-1] / 2 + image[i, j] / 2).astype(int)
                    output[i, j+1] = image[i, j]
                    output[i, j+2:] = image[i, j+1:]
            image = output
            w += 1
            count += 1
            
    return output

def add_horizontal_seam(image, seams):
    """
    Add a horizontal seam to the image by transposing.
    :param image: Input image (H, W, 3) or (H, W)
    :param seam: List of row indices (length W)
    :return: Image with seam added (H+1, W, 3) or (H+1, W)
    """
    if len(image.shape) == 3:
        return add_vertical_seam(np.transpose(image, (1, 0, 2)), seams).transpose(1, 0, 2)
    else:
        return add_vertical_seam(image.T, seams).T


def seam_carve(image, delta_width, delta_height):
    """
    Perform seam carving to resize the image to target dimensions.
    :param image: Input image (H, W, 3)
    :param delta_width: width difference to resize
    :param delta_height: height difference to resize
    :return: Resized image (h + delta_height, w + delta_width, 3)
    """
    img = np.array(image)
    if img.shape[2] > 3:
        img = img[:, :, :3]
        print("Num_channel > 3, only using first 3 channels")
    
    # Alternate between horizontal and vertical operations
    alternate = True
    
    if delta_height > 0:
        energy = compute_energy(img)
        seams = find_horizontal_seam(energy, delta_height)
        img = add_horizontal_seam(img, seams)
        delta_height = 0
    elif delta_width > 0:
        energy = compute_energy(img)
        seams = find_vertical_seam(energy, delta_width)
        img = add_vertical_seam(img, seams)
        delta_width = 0
        
    while delta_width != 0 or delta_height != 0:
        print(f"({delta_width}, {delta_height})")
        if delta_width != 0 and (delta_height == 0 or (alternate and delta_height != 0)):
            # Vertical operation
            energy = compute_energy(img)
            seam = find_vertical_seam(energy)[0]
            
            img = remove_vertical_seam(img, seam)
            delta_width += 1
            
            alternate = False if delta_height != 0 else True
        
        elif delta_height != 0:
            # Horizontal operation
            energy = compute_energy(img)
            seam = find_horizontal_seam(energy)[0]
            
            img = remove_horizontal_seam(img, seam)
            delta_height += 1
            
            alternate = True if delta_width != 0 else False
    
    return Image.fromarray(np.uint8(img))
