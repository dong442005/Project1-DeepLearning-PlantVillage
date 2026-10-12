import os
import shutil
import kagglehub
from sklearn.model_selection import train_test_split

# a: tải dataset về rồi chia train/val/test, chạy 1 lần thôi
# sau khi chia xong thì giữ nguyên partition để mọi người train cùng split

print("Đang tải PlantVillage Dataset từ Kaggle...")
dataset_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print(f"Dataset đã tải xong tại: {dataset_path}")

# bộ này có 3 thư mục con: color, segmented, grayscale — mình dùng color
source_dir = os.path.join(dataset_path, "plantvillage dataset", "color")
if not os.path.exists(source_dir):
    source_dir = dataset_path  # phòng khi kaggle đổi cấu trúc

print(f"Thư mục ảnh gốc: {source_dir}")

# data/ nằm ở thư mục gốc repo, script này chạy từ src/ nên phải ra ngoài 1 cấp
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# xóa data cũ trước khi chia lại, tránh ảnh cũ lẫn vào
if os.path.exists(base_dir):
    print("Xóa thư mục data cũ...")
    shutil.rmtree(base_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
print(f"\nTìm thấy {len(classes)} lớp.")
print("Bắt đầu chia 70% train - 15% val - 15% test")

for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    cls_path = os.path.join(source_dir, cls)
    images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.JPG', '.png', '.jpeg'))]

    # chia 2 lần: lần 1 lấy 70% train, lần 2 chia đôi 30% còn lại thành val/test
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    def copy_files(img_list, dest_folder):
        for img in img_list:
            shutil.copy(os.path.join(cls_path, img), os.path.join(dest_folder, img))

    copy_files(train_imgs, os.path.join(train_dir, cls))
    copy_files(val_imgs, os.path.join(val_dir, cls))
    copy_files(test_imgs, os.path.join(test_dir, cls))

print("==========================================")
print("✅ Chia xong! Data nằm trong thư mục data/")
print("==========================================")

# ==========================================
# đoạn mẫu dưới đây dành cho b, c, d copy vào file của mình
# (hoặc import thẳng vào notebook)
# ==========================================
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# z-score dùng mean/std tính trên tập train (ko đụng vào val/test)
def z_score_norm(img):
    img = img / 255.0
    mean = np.array([0.46658055, 0.48930454, 0.41043331])
    std  = np.array([0.1992171,  0.17497431, 0.21742669])
    return (img - mean) / std

# lưu ý b và d dùng transfer learning thì ko dùng hàm này
# thay bằng preprocess_input của model tương ứng
# vd: from tensorflow.keras.applications.resnet50 import preprocess_input

# augment chỉ cho train, val/test chỉ normalize thôi
train_datagen = ImageDataGenerator(
    preprocessing_function=z_score_norm,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
test_val_datagen = ImageDataGenerator(preprocessing_function=z_score_norm)

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
    shuffle=False  # phải tắt shuffle ở test để confusion matrix ko bị lệch
)

# class weight vì data lệch nhau khá nhiều giữa các lớp
train_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights_array))

# nhớ truyền class_weight_dict vào model.fit():
# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=val_generator,
#     class_weight=class_weight_dict  # quan trọng
# )
'''
