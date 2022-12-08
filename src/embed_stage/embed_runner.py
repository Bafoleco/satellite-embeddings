import sys
sys.path.append('../')

import embed_stage.embed as embed

# model_name = "pretrained_vgg13_ElRdInNl"
model_name = "pretrained_visiontransformer_1024_ElRdInTr"
embed.save_embeddings(model_name)

