'''
CONFIGURATION SETTINGS FOR PROJECT
'''
import torch

cfg = {
    'DATA': {
        'TRAIN_PATH': 'data/image_names.pkl',
        'TRAIN_TEXT_TOKEN_PATH': 'data/embeded_dicom_fixed.pkl',
        'VAL_PATH': 'data/p10_image_names.pkl',
        'VAL_TEXT_TOKEN_PATH': 'data/p10_embeded_dicom_fixed.pkl'
    },
    'CALLBACK': {
        'dirpath': 'results/all_maskloss',
        'save_top_k': 2,
        'monitor': 'val_loss'
    },
    'LOGGER': {
        'save_dir': 'results/',
        'name': 'imglogs'
    },
    'TRAIN': {
        'model_def': 'ModelV2',
        'TRAINER': {
            'max_epochs': 100,
            'precision': 16,
            'enable_checkpointing': True,
            'accelerator': 'gpu',
            'devices': int(torch.cuda.device_count()),
            'strategy': 'ddp_find_unused_parameters_true'
        },
        'LR': 1e-5,
        'BATCH_SIZE': 16,
        'DATA_PERC': 1.0  # Float between 0 and 1 for how much of total train-val data to use
    },
    'MODEL': {
        'ModelV1': {
            'VIT': {
                'embed_dim': 2048,
                'hidden_dim': 512,
                'n_heads': 4,
                'n_layers': 4,
                'dropout': 0.3,
                'attention_type': 'fast',
                'n_features': 256
            },
            'PROJECTOR': {
                'img_embed_dim': 2048,
                'txt_embed_dim': 768,
                'out_dim': 256
            },
            'DECODER': {
                'outchn': 1,
                'in_ft': 1,
                'img_size': 256,
                'apply_attention': True,
                'emb_dim': 256,
                'nheads': 4,
            }
        },
        'ModelV2': {
            'ENCODER': {
                'model_path': 'data/pretrained_weights/vqgan/last.ckpt',
                'cfg_path': 'data/pretrained_weights/vqgan/2021-12-17T08-58-54-project.yaml',
                'codebook_path': 'data/pretrained_weights/vqgan/mimiccxr_vqgan1024_res512_codebook_indices.pickle'
            },
            'DECODER': {
                'embed_dim': 1024,
                'hidden_dim': 4096,
                'n_heads': 8,
                'n_layers': 6,
                'dropout': 0.1,
                'attention_type': 'normal',
                'n_features': 256
            },
            'PROJECTOR': {
                'txt_embed_dim': 768,
            }
        }
    },
    'WANDB': {
        "MODE": 'online'  # One of {'online', 'offline', 'disabled'}
    }
}
