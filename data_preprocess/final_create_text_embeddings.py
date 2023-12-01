from data_preprocess.text_json import create_json
import json
import pickle
from data_preprocess.reports_only_clean import get_common_view
from src.models.new_bert_txt_encoder import get_text_embeddings, get_cxr_bert_tokenizer_and_encoder

def create_text_embeddings(raw_texts_root, json_output_file, cleaned_json_outputfile, pkl_outputfile, metadata_csv_file):
    """
    Generate a cleaned dictionary of dicom IDs and corresponding texts. 

    <raw_texts_root>: Location of folder with all raw text
    <json_output_file>: file location to output dictionary of dicom IDs and text that have been split to include interpretations and findings
    <cleaned_json_outputfile>: JSON file with the dictionary of Dicom IDs and text that correspond to the image with the most common view (AP) 
    <pkl_outputfile>: a .pickle file with the dictionary with keys as the dicom ID and the embedding of text of shape [1, 256, 2]
    which are the stacked input_ids and attention_mask
    <metadata_csv_file>: the mimic-cxr-2.0.0-metadata.csv file location which is used to identify the most common view of the CXR images used 
    for data cleanin
    """
    # create json of text file paths in a list 
    create_json(raw_texts_root, json_output_file)
    print("got text path JSON")
    # get common view, get the actual text, and split to get interpretations
    get_common_view(metadata_csv_file, json_output_file, cleaned_json_outputfile)
    print(f"cleaned for common view and retreived interpretation, see texts at {cleaned_json_outputfile}")
    tokenizer, model = get_cxr_bert_tokenizer_and_encoder()


    with open(cleaned_json_outputfile, "r") as file:
        text_dict = json.load(file)

    for dicom in list(text_dict.items()):
        # get a dictionary with dicom IDs as keys and embedding of interpretatio section [1, 256, 2] as values
        stacked_tensor = get_text_embeddings([text_dict[dicom[0]]], tokenizer, model)
        text_dict[dicom[0]] = stacked_tensor
        #print(stacked_tensor.size())
    # dump dict to pickel
    with open(pkl_outputfile, "wb") as file:
        pickle.dump(text_dict, file)
    print(f"created pickel for dicom ID and text embeddings, download pickle file at {pkl_outputfile}")

if __name__ == "__main__":
    raw_texts_root = 'data/downloaded_reports/physionet.org/files/mimic-cxr'
    json_output_file = 'data/final_raw_text.json'
    cleaned_json_outputfile = 'data/final_cleaned_text.json'
    pkl_outputfile = 'data/final_cleaned_embeddings.pkl'
    metadata_csv_file = 'data/mimic-cxr-2.0.0-metadata.csv'
    create_text_embeddings(raw_texts_root, json_output_file, cleaned_json_outputfile, pkl_outputfile, metadata_csv_file)



