import numpy as np

# Load landmark file (text format with x y coordinates)
lms = []
with open('minimal_server/models/sanders/landmarks/0.lms', 'r') as f:
    for line in f:
        x, y = line.strip().split()
        lms.append([float(x), float(y)])

lms = np.array(lms)

print(f"Type: {type(lms)}")
print(f"Shape: {lms.shape}")
print(f"\nFirst 10 landmarks:")
print(lms[:10])

# Extract ROI bounds
xs = lms[:, 0]
ys = lms[:, 1]
print(f"\nX range: {xs.min()} to {xs.max()}")
print(f"Y range: {ys.min()} to {ys.max()}")

x1, y1 = int(xs.min()), int(ys.min())
x2, y2 = int(xs.max()), int(ys.max())
width = x2 - x1
height = y2 - y1

print(f"\nROI bounds: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
print(f"ROI size: {width}x{height}")
