import os
import shutil
import kagglehub
from sklearn.model_selection import train_test_split

# ==========================================
# PHẦN 1: TẢI VÀ CHIA TẬP DỮ LIỆU (Thành viên A chạy 1 lần duy nhất)
# ==========================================

print("Đang tải PlantVillage Dataset từ Kaggle...")
dataset_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print(f"Dataset đã tải xong tại: {dataset_path}")

# Kiểm tra cấu trúc thư mục gốc (Thường bộ này có thư mục 'plantvillage dataset/color' hoặc 'plantvillage dataset/segmented')
# Ở đây ta ưu tiên dùng ảnh gốc có màu ('color')
source_dir = os.path.join(dataset_path, "plantvillage dataset", "color")
if not os.path.exists(source_dir):
    source_dir = dataset_path # Đề phòng cấu trúc file thay đổi

print(f"Thư mục chứa ảnh gốc: {source_dir}")

# Thư mục đích trong Project của chúng ta(thư mục data/)
# Lưu ý: file này đang ở trong thư mục src/, nên thư mục gốc là '../'
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Xóa thư mục cũ nếu có để tránh dữ liệu rác
if os.path.exists(base_dir):
    print("Xóa thư mục data cũ...")
    shutil.rmtree(base_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
print(f"\nTìm thấy {len(classes)} loại lá/bệnh (classes).")
print("Bắt đầu sao chép và chia tập Train(70%) - Val(15%) - Test(15%)... (Có thể mất vài phút)")

for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    
    cls_path = os.path.join(source_dir, cls)
    # Lấy toàn bộ ảnh của MỘT lớp
    images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.JPG', '.png', '.jpeg'))]
    
    # Chia lần 1: 70% Train, 30% cho (Val + Test)
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    # Chia lần 2: Lấy 30% chia đôi -> 15% Val, 15% Test
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    # Hàm copy ẩn đi để code gọn
    def copy_files(img_list, dest_folder):
        for img in img_list:
            shutil.copy(os.path.join(cls_path, img), os.path.join(dest_folder, img))
            
    copy_files(train_imgs, os.path.join(train_dir, cls))
    copy_files(val_imgs, os.path.join(val_dir, cls))
    copy_files(test_imgs, os.path.join(test_dir, cls))

print("==========================================")
print("✅ HOÀN TẤT PHÂN CHIA DỮ LIỆU!")
print("Dữ liệu đã nằm gọn trong thư mục data của Project.")
print("==========================================")

# ==========================================
# PHẦN 2: CHUẨN BỊ GENERATOR CHO CÁC MÔ HÌNH (Dùng chung cho Thành viên B, C, D)
# ==========================================
"""
ĐOẠN CODE MẪU BÊN DƯỚI DÀNH CHO B, C, D COPY VÀO FILE CODE CỦA MÌNH
(Hoặc import trực tiếp vào file .ipynb)
"""
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Định nghĩa các hằng số
IMG_SIZE = (224, 224) # Kích thước ảnh chuẩn hóa (Khuyên dùng cho ResNet/VGG/MobileNet)
BATCH_SIZE = 32

# 1. Hàm chuẩn hóa Z-Score dựa trên Mean/Std tính được từ tập Train
import numpy as np

def z_score_norm(img):
    img = img / 255.0
    mean = np.array([0.46658055, 0.48930454, 0.41043331])
    std = np.array([0.1992171,  0.17497431, 0.21742669])
    return (img - mean) / std

# LƯU Ý CHO THÀNH VIÊN A và D (Transfer Learning): 
# Thay vì dùng hàm z_score_norm này, hãy import và dùng hàm của chính model đó.
# Ví dụ: from tensorflow.keras.applications.resnet import preprocess_input

# 2. Khởi tạo Data Augmentation (Tăng cường dữ liệu) chỉ cho tập Train
train_datagen = ImageDataGenerator(
    preprocessing_function=z_score_norm, # Dùng Z-Score thay cho rescale=1./255
    rotation_range=20,       # Xoay ngẫu nhiên
    width_shift_range=0.2,   # Dịch chuyển ngang
    height_shift_range=0.2,  # Dịch chuyển dọc
    horizontal_flip=True,    # Lật ngang ảnh
    zoom_range=0.2           # Thu phóng ngẫu nhiên
)

# Tập Validation và Test KHÔNG ĐƯỢC làm méo, chỉ chuẩn hóa pixel bằng Z-score
test_val_datagen = ImageDataGenerator(preprocessing_function=z_score_norm)

# 2. Đọc dữ liệu từ thư mục (Flow from directory)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = test_val_datagen.flow_from_directory(
    'data/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # [QUAN TRỌNG] Phải tắt shuffle ở tập Test để vẽ Confusion Matrix không bị sai lệch
)

# 3. Tính toán Class Weights (Trọng số mất cân bằng) tự động
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

train_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Chuyển đổi sang dạng dictionary (từ điển) vì Keras yêu cầu định dạng này
class_weight_dict = dict(enumerate(class_weights_array))

# LƯU Ý KHI TRAIN (DÀNH CHO B, C, D): 
# Hãy nhớ nhét biến class_weight_dict vào tham số class_weight trong hàm model.fit()
# Ví dụ:
# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=val_generator,
#     class_weight=class_weight_dict  <--- CỰC KỲ QUAN TRỌNG
# )
'''
