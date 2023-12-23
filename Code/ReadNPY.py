import numpy as np

# Specify the path to your .npy file
file_path = path_neutral_star = '/STAR/Code/manipulated_star_poses.npy'

# Load the .npy file
data = np.load(file_path)

# Now, 'data' contains the contents of the .npy file
print(data)