import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from pathlib import Path
from detector.utils.image import load_image_rgb as load_image


class EmbeddingGenerator:
    """Load a vision model and processor once and generate normalized image embeddings.
"""

    def __init__(self, model: str = "google/siglip2-base-patch16-512", device_map="auto"):
        self.model = model
        # load model+processor once
        self.model = AutoModel.from_pretrained(model, device_map=device_map).eval()
        self.processor = AutoProcessor.from_pretrained(model)

    def embed_from_path(self, path: str):
        p = Path(path)
        
        image = load_image(str(p))
        inputs = self.processor(images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
            emb = F.normalize(image_embeddings, p=2, dim=-1)
            prepared = emb.squeeze(0).float().cpu()
        return prepared.tolist()

    def embed_from_pil(self, image):
        """Embed a PIL image object."""
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
            emb = F.normalize(image_embeddings, p=2, dim=-1)
            prepared = emb.squeeze(0).float().cpu()
        return prepared.tolist()
