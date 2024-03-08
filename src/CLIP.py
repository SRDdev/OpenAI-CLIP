import torch 
from torch import nn
from torch.nn import functional as F

import config as config
from module import ImageEncoder,TextEncoder,ProjectionHead

#-------------------CLIP------------------#
class CLIPModel(nn.Module):
    """
    Implementation of CLIP Architecture.
    Args:
        Input  :
        Output :
    Note :
    """
    def __init__(self,temperature=config.temperature,image_embedding=config.image_embedding,text_embedding=config.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.text_proj = ProjectionHead(embedding_dim=text_embedding)
        self.image_proj = ProjectionHead(embedding_dim=image_embedding)
        self.temperature = temperature
    
    def forward(self,batch):
        # Get Image and Text features
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(input_ids=batch["input_ids"],attention_mask=batch['attention_mask'])

        # Getting the Image and Text features to same dim.
        image_embeddings = self.image_proj(image_features)
        text_embeddings = self.text_proj(text_features)

        # Contrastive Loss (@ used for matrix multiplication)
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        image_similarity = image_embeddings @ image_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (image_similarity + text_similarity)/(2*self.temperature) , dim=1 
            )
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T,targets.T,reduction='none')
        loss = ((images_loss+texts_loss)/2.0)
        return loss.mean()
    
# #----------------------Main-----------------#
# if __name__ == '__main__':
#     images = torch.randn(8, 3, 224, 224)
#     input_ids = torch.randint(5, 300, size=(8, 25))
#     attention_mask = torch.ones(8, 25)
#     batch = {
#         'image': images,
#         'input_ids': input_ids,
#         'attention_mask': attention_mask
#     }

#     CLIP = CLIPModel()
#     loss = CLIP(batch)
#     print("CLIP")