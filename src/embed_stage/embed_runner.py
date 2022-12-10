import sys
sys.path.append('../')

import embed_stage.embed as embed

# model_name = "pretrained_vgg13_ElRdInNl"
model_name = "baseline_model"
embed.save_embeddings(model_name)

# model_name = "pretrained_visiontransformer_2048_ElRdInTr"
# embed.save_embeddings(model_name)

# model_name = "pretrained_visiontransformer_4096_ElRdInTr"
# embed.save_embeddings(model_name)
