# File MobileNetV2 được lưu bằng Keras mới hơn môi trường đang dùng.
import json
from pathlib import Path
import shutil
import tempfile


def load_saved_model(path):
    """Đọc H5 từ Keras mới có metadata chưa được Keras 3.13 hỗ trợ."""
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(path, compile=False)
    except TypeError as error:
        if path.suffix.lower() not in {".h5", ".hdf5"} or "input_axes" not in str(error):
            raise
        import h5py

        def clean_metadata(value):
            changed = 0
            if isinstance(value, dict):
                if value.get("class_name") == "BatchNormalization":
                    config = value.get("config", {})
                    if config.get("renorm") is False:
                        # Renormalization bị tắt khi train: các tùy chọn này không dùng.
                        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
                            if key in config:
                                del config[key]
                                changed += 1
                if value.get("module") == "keras.initializers":
                    config = value.get("config", {})
                    for key in ("input_axes", "output_axes"):
                        if key in config:
                            if config[key] is not None:
                                raise ValueError("Không hỗ trợ initializer có axes khác None") from error
                            del config[key]
                            changed += 1
                for child in value.values():
                    changed += clean_metadata(child)
            elif isinstance(value, list):
                for child in value:
                    changed += clean_metadata(child)
            return changed

        # Chỉ sửa metadata trên bản sao tạm; toàn bộ trọng số và file gốc giữ nguyên.
        with tempfile.TemporaryDirectory(prefix="plantvillage-model-") as temp:
            compatible = Path(temp) / path.name
            shutil.copyfile(path, compatible)
            with h5py.File(compatible, "r+") as handle:
                config = json.loads(handle.attrs["model_config"])
                changed = clean_metadata(config)
                if not changed:
                    raise error
                handle.attrs["model_config"] = json.dumps(config)
            print("  Nạp bản sao tương thích metadata Keras (trọng số giữ nguyên).", flush=True)
            return tf.keras.models.load_model(compatible, compile=False)
