from PIL import Image

file_path = "../data/physionet.org/files/mimic-cxr-jpg\\2.0.0\\files\\p10\\p10000032\\s53911762\\68b5c4b1-227d0485-9cc38c3f-7b84ab51-4b472714.jpg"

try:
    with Image.open(file_path) as img:
        img.show()  
    
except IOError:
    print("Error in opening the image.")