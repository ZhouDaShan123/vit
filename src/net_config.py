from easydict import EasyDict as edict


# Returns the ViT-B/16 configuration.
get_b16_config = edict({
    'patches_grid': None,
    'patches_size': 16,
    'hidden_size': 768,
    'transformer_mlp_dim': 3072,
    'transformer_num_heads': 12,
    'transformer_num_layers': 12,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 1.0,  # 0.9
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-B/32 configuration.
get_b32_config = edict({
    'patches_grid': None,
    'patches_size': 32,
    'hidden_size': 768,
    'transformer_mlp_dim': 3072,
    'transformer_num_heads': 12,
    'transformer_num_layers': 12,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-L/16 configuration.
get_l16_config = edict({
    'patches_grid': None,
    'patches_size': 16,
    'hidden_size': 1024,
    'transformer_mlp_dim': 4096,
    'transformer_num_heads': 16,
    'transformer_num_layers': 24,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-L/32 configuration.
get_l32_config = edict({
    'patches_grid': None,
    'patches_size': 32,
    'hidden_size': 1024,
    'transformer_mlp_dim': 4096,
    'transformer_num_heads': 16,
    'transformer_num_layers': 24,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-L/16 configuration.
get_h14_config = edict({
    'patches_grid': None,
    'patches_size': 14,
    'hidden_size': 1280,
    'transformer_mlp_dim': 5120,
    'transformer_num_heads': 16,
    'transformer_num_layers': 32,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})
