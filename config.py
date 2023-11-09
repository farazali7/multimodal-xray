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
            'DECODER': {
                'outchn': 512,
                'in_ft': 2048,
                'img_size': 256,
                'apply_attention': True,
                'emb_dim': 512,
                'nheads': 4,
            }
        }
    }
}
