import numpy as np

# Specify the path to your .npy file
file_path = path_neutral_star = 'C:/Users/tfran/Desktop/Inzynierka/STAR/STAR/Code/star_poses.npy'

# Load the .npy file
data = np.load(file_path)

# Now, 'data' contains the contents of the .npy file
print(data)