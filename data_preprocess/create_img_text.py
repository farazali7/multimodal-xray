import json
import os 

def doit():
    with open("test.json", "r") as file:
        data = json.load(file)

    image_paths = data["images"]
    path_img_dict = {}
    for img in image_paths:
        study_id = img.split("/")[-2]
        patient_id = img.split("/")[-3]
        img_small = patient_id + '/' + study_id
        path_img_dict.setdefault(img_small, []).append(img)  

    with open("1.json", "w") as f:
        json.dump(path_img_dict, f, indent=4)


        

    path_txt_dict = {}
    for root, _, files in os.walk("../data/downloaded_reports/physionet.org/files/p10"):
            
            for file in files:
                if file.endswith('.txt'):
                    path = os.path.join(root, file)
                    patient_id = path.split(os.sep)[-2]
                    study_id = path.split(os.sep)[-1].split('.')[0]
                    txt_small = patient_id + '/' + study_id
                    path_txt_dict[txt_small] = path  
    with open("2.json", "w") as f:
        json.dump(path_txt_dict, f, indent=4)
    


    
    path_label_dict = {}
    for short_path in path_img_dict.keys():
        if short_path in path_txt_dict:
            #{"full length path": label}
            for path in path_img_dict[short_path]:
                path_label_dict[path] = path_txt_dict[short_path]
    #print(path_label_dict.keys())
    with open("3.json", "w") as f:
        json.dump(path_label_dict, f, indent=4)



    path_label_dict_text = {key: read_file(value) for key, value in path_label_dict.items()}

    print(len(path_label_dict_text))

    with open("destination_path.json", "w") as f:
        json.dump(path_label_dict_text, f, indent=4)

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    impression = content.split(":")[-1].strip()

    return impression


if __name__ == "__main__":
    doit()


