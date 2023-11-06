# Sentence Ax


![Sentence Ax](pics/sentence_ax_logo.jpg)

Sentence Ax is a complete rewrite, from stem to stern, of Openie6.

https://github.com/dair-iitd/openie6

Sentence Ax decomposes a compound or complex sentence
into a set of simple sentences (extractions). It does this using 
a fine-tuned BERT model.

Sentence Ax is a stand-alone app, but 
it's also 
a key component of the 
Mappa Mundi Project.
Both Sentence Ax and Mappa Mundi were
written by
www.ar-tiste.xyz

## Input Dataset and Weights

Due to a < 50MB per file limitation at Github, the  
input datasets and weights will be located at HuggingFace.

* input dataset. Unzip [this file](https://huggingface.co/datasets/rrtucci/SentenceAx-input-data) and use it to replace the 
directory `input_data` of this github repo. 

## History

[This blog post](https://qbnets.wordpress.com/2023/11/05/sentenceax-my-open-source-software-for-sentence-splitting/)
 gives more info about SentenceAx,
at the time (2023/11/05) of its first public unveiling.
