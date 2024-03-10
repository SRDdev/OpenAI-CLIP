# OpenAI CLIP: Contrastive Language-Image Pre-training

This repository is an educational attempt to understand and replicate OpenAI CLIP's 2021 model in PyTorch.

```
Note: OpenAI has open-sourced code related to the CLIP model, but it is complicated to understand.
I prefer to implement it based on the paper. Additionally, I found a helpful tutorial inspired by the
CLIP model on Keras code examples, which assisted me in replicating the code in PyTorch.
```

---

OpenAI introduced the CLIP model in 2021, taking a backseat to the prominence of DALL-E. This project delves into the process of building the CLIP model from scratch using PyTorch, offering a simplified approach compared to the initially complex and extensive code released by OpenAI. Inspired by a Keras code examples tutorial, relevant sections have been translated into PyTorch, presenting a concise and accessible tutorial tailored for the PyTorch framework.

![CLIP](https://production-media.paperswithcode.com/methods/3d5d1009-6e3d-4570-8fd9-ee8f588003e7.png)

## What is CLIP?
In the paper titled "Learning Transferable Visual Models From Natural Language Supervision," OpenAI introduces CLIP, short for Contrastive Language-Image Pre-training. This model learns how sentences and images are related, retrieving the most relevant images for a given sentence during training. What sets CLIP apart is its training on complete sentences instead of individual categories like cars or dogs. This approach allows the model to learn more and discover patterns between images and text. When trained on a large dataset of images and their corresponding texts, CLIP can also function as a classifier, outperforming models trained directly on ImageNet for classification tasks. Further exploration of the paper reveals in-depth details and astonishing outcomes.

## Why is CLIP Important?
CLIP is significant due to its groundbreaking approach to learning visual representations through natural language supervision. By understanding relationships between images and full sentences, it surpasses traditional models, exhibiting versatility across tasks such as image classification, object detection, and text comprehension. The model's transfer learning capability, acquired through training on a diverse dataset, makes it adaptable to various downstream applications. CLIP's increased application is driven by OpenAI's provision of pre-trained models, enhancing accessibility for developers. Its superior performance on benchmark datasets, coupled with its open-source nature, has encouraged widespread adoption and integration into diverse projects. The model's elimination of siloed training and its ability to transfer knowledge seamlessly between modalities further fuel its application across a spectrum of tasks and domains.

## How to Use
1. **Installation:** Clone the repository and install the required dependencies.
    ```bash
    git clone https://github.com/SRDdev/OpenAI-CLIP.git
    cd OpenAI-CLIP
    pip install -r requirements.txt
    ```
2. **Training:** Follow the provided code and instructions to train the CLIP model from scratch.
    ```bash
    python src/train.py
    ```

3. **Inference:** Utilize the openai-clip.ipynb for inference code.

Feel free to explore the code, modify it according to your needs, and contribute to enhancing this educational project.

## Acknowledgments
This implementation is inspired by OpenAI's CLIP model and Keras code examples.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
