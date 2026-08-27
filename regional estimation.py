import random
import torch
import scipy.io as sio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from models import MLP  # Assuming your MLP model is defined in 'models.py'
from copy import deepcopy

# Set random seeds for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Load data
date = '0510'
data1 = sio.loadmat(f'../data/SAR_DATA/new_data/{date}/c3.mat')['data']
data2 = sio.loadmat(f'../data/SAR_DATA/new_data/{date}/freeman.mat')['data']
data3 = sio.loadmat(f'../data/SAR_DATA/new_data/{date}/h_a_alpha.mat')['data']
data4 = sio.loadmat(f'../data/SAR_DATA/new_data/{date}/nned.mat')['data']
data5 = sio.loadmat(f'../data/SAR_DATA/new_data/{date}/tsvm.mat')['data']
data6 = sio.loadmat(f'../data/SAR_DATA/new_data/{date}/yamaguchi.mat')['data']
datas = np.concatenate((data1, data2, data3, data4, data5, data6), axis=2)
min_val = np.min(datas, axis=0)  # 每列的最小值
max_val = np.max(datas, axis=0)  # 每列的最大值
datas = (datas - min_val) / (max_val - min_val)  # 归一化
binary_mask = np.load(f'../data/wheat_binary_mask_{date}.npy')

# Device setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model instances and load weights
model1 = MLP()  # Assuming MLP is a class you defined
model2 = MLP()
model1.init_weights()
model2.init_weights()
model1.load_state_dict(torch.load(f'./model1_weights_{date}.pth',  map_location=torch.device('cpu')))
model2.load_state_dict(torch.load(f'./model2_weights_{date}.pth', map_location=torch.device('cpu')))
model1.to(device)
model2.to(device)
model1.eval()
model2.eval()

# Apply model predictions to the masked region
def get_model_prediction(data, model):
    # Convert the masked data to a tensor and pass it through the model
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():  # Inference mode, no gradient calculation
        model_output, features = model(data_tensor)
        print(model_output)
    return model_output.cpu().numpy()

# Get predictions from both models for the masked data
pred1 = get_model_prediction(datas, model1)
pred2 = get_model_prediction(datas, model2)

# Average the results of both models
final_prediction = np.mean([pred1, pred2], axis=0)

# Reshape the final prediction back to the original image shape
final_prediction = np.reshape(final_prediction, binary_mask.shape)
print(final_prediction.shape, binary_mask.shape)

# Initialize the final image with zeros (black for non-masked areas)
estimation_image = np.zeros_like(binary_mask, dtype=np.float32)
# Assign predictions to the masked areas
estimation_image[binary_mask == 1] = final_prediction[binary_mask == 1]
estimation_image[binary_mask == 0] = 0

# Clip LAI values to [0, 7]
estimation_image = np.clip(estimation_image, 0, 7)

# Convert to uint8 for proper visualization
img = Image.fromarray((estimation_image).astype(np.uint8)).convert('L')

# 目标长宽比
target_ratio = 6 / 5.2

# 计算新的尺寸
width, height = img.size
if width / height > target_ratio:
    new_width = int(height * target_ratio)
    new_height = height
else:
    new_width = width
    new_height = int(width / target_ratio)

# 调整图片尺寸（等比例缩放后填充到目标比例）
img_resized = img.resize((new_width, new_height), Image.NEAREST)

# 逆时针旋转 9 度
img_rotated = img_resized.rotate(9, expand=True, resample=Image.NEAREST)
final_image = deepcopy(np.asarray(img_rotated))
h, w = final_image.shape[0], final_image.shape[1]
for i in range(h):
    for j in range(w):
        if final_image[i][j] < 3:
            final_image[i, j] = 0

final_image = Image.fromarray(final_image.astype(np.uint8)).convert('L')

# 定义自定义颜色映射：从白色到黄色再到绿色
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'green', 'blue'])

# Visualize the result using the custom colormap, 设置vmin=0, vmax=7使得低值显示为白色，高值显示为绿色，中间过渡为黄色
plt.imshow(final_image, cmap=custom_cmap, vmin=4, vmax=7)
plt.colorbar()  # Add color bar to indicate intensity

if date == '0416':
    plt.title('April 16')
elif date == '0510':
    plt.title('May 10')
else:
    plt.title('June 3')

# Remove axis labels and ticks
plt.axis('off')

# Save the final image
plt.savefig(f'./regional_prediction_{date}.png', bbox_inches='tight', pad_inches=0)
plt.show()

print('max:', np.max(final_prediction[binary_mask == 1]))
print('min:', np.min(final_prediction[binary_mask == 1]))
