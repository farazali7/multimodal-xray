'''
CONFIGURATION SETTINGS FOR PROJECT
'''

cfg = {
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
        }
    }
}
