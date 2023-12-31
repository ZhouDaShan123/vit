import copy

import mindspore
from mindspore import Parameter, Tensor, ops
import mindspore.nn as nn


def swish(x):
    return x * ops.sigmoid(x)


class Attention(nn.Cell):
    """Attention"""
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer_num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.attention_head_size2 = Tensor(config.hidden_size / self.num_attention_heads, mindspore.float32)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.out = nn.Dense(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(p=1 - config.transformer_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(p=1 - config.transformer_attention_dropout_rate)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        """transpose_for_scores"""
        new_x_shape = ops.shape(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = ops.reshape(x, new_x_shape)
        return ops.transpose(x, (0, 2, 1, 3,))

    def construct(self, hidden_states):
        """construct"""
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = mindspore.ops.matmul(query_layer, ops.transpose(key_layer, (0, 1, 3, 2)))
        attention_scores = attention_scores / ops.sqrt(self.attention_head_size2)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = mindspore.ops.matmul(attention_probs, value_layer)
        context_layer = ops.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = ops.shape(context_layer)[:-2] + (self.all_head_size,)
        context_layer = ops.reshape(context_layer, new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Cell):
    """Mlp"""
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Dense(config.hidden_size, config.transformer_mlp_dim,
                            weight_init='XavierUniform', bias_init='Normal')
        self.fc2 = nn.Dense(config.transformer_mlp_dim, config.hidden_size,
                            weight_init='XavierUniform', bias_init='Normal')
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=1 - config.transformer_dropout_rate)

    def construct(self, x):
        """construct"""
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Cell):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None

        patch_size = config.patches_size
        n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.hybrid = False

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size, has_bias=True)
        self.position_embeddings = Parameter(ops.zeros((1, n_patches+1, config.hidden_size), mindspore.float32),
                                             name="q1", requires_grad=True)
        self.cls_token = Parameter(ops.zeros((1, 1, config.hidden_size), mindspore.float32), name="q2",
                                   requires_grad=True)

        self.dropout = nn.Dropout(p=1 - config.transformer_dropout_rate)

    def construct(self, x):
        """construct"""
        B = x.shape[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (B, self.cls_token.shape[1], self.cls_token.shape[2]))

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = ops.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = ops.transpose(x, (0, 2, 1))
        x = ops.cat((cls_tokens, x), axis=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Cell):
    """Block"""
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)
        self.ffn_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def construct(self, x):
        """construct"""
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Cell):
    """Encoder"""
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.CellList([])
        self.encoder_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)
        for _ in range(config.transformer_num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def construct(self, hidden_states):
        """construct"""
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Cell):
    """Transformer"""
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def construct(self, input_ids):
        """construct"""
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded


class VisionTransformer(nn.Cell):
    """VisionTransformer"""
    def __init__(self, config, img_size=(224, 224), num_classes=21843):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size)
        self.head = nn.Dense(config.hidden_size, num_classes)

    def construct(self, x, labels=None):
        """construct"""
        x = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits
