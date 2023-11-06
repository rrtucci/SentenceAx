# Sentence Ax


![Sentence Ax](pics/sentence_ax_logo.jpg)

Sentence Ax is a complete rewrite, from stem to stern, of Openie6.

https://github.com/dair-iitd/openie6

Sentence Ax decomposes a compound or complex sentence
into a set of simple sentences (extractions). It does this using 
a fine-tuned BERT model.

Sentence Ax is a stand-alone app, but, just like the
SCuMpy app,
it's also 
a key component of the 
Mappa Mundi Project which started with the
MappaMundi app.
The SentenceAx app, SCuMpy app and Mappa Mundi app were
written by
www.ar-tiste.xyz

* [Mappa Mundi Project](https://qbnets.wordpress.com/2023/07/31/searching-for-causal-pathways-for-diseases-using-an-individuals-fitbit-and-social-media-records-part-2/)
    * [MappaMundi app](https://github.com/rrtucci/mappa_mundi)
    * [SCuMpy app](https://github.com/rrtucci/scumpy)
    * SentenceAx (this repo)
    * Causal Fitbit (coming soon)


## Weights and Input Dataset

Due to a < 50MB per file limitation at Github, the weights and input datasets will be located at HuggingFace.

* input dataset. Unzip [this file](https://huggingface.co/datasets/rrtucci/SentenceAx-input-data) and use it to replace the 
directory `input_data` of this github repo. 

## History

[This blog post](https://qbnets.wordpress.com/2023/11/05/sentenceax-my-open-source-software-for-sentence-splitting/)
 gives more info about SentenceAx,
at the time (2023/11/05) of its first public unveiling.
