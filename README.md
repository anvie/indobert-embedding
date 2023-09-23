
Text embedding encoder using [indolem/indobert-base-uncased](https://huggingface.co/indolem/indobert-base-uncased) as the model.

## Installation

```bash
pip install indobert-embedding
```

## Usage

```python
from indo_bert_embedding import get_embedding

embedding = get_embedding("Saya belajar NLP di Neuversity.")
```

For get text similarity distance:

```python
from indo_bert_embedding import text_similarity

distance = text_similarity("Saya belajar NLP di Neuversity.", "Aku belajar NLP di Universitas Indonesia.")
```

`text_similarity` using cosine similarity to calculate distance.

## Citation

```
@inproceedings{koto2020indolem,
  title={IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP},
  author={Fajri Koto and Afshin Rahimi and Jey Han Lau and Timothy Baldwin},
  booktitle={Proceedings of the 28th COLING},
  year={2020}
}
```