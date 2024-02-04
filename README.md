# SentenceAx


![SentenceAx](pics/sentence_ax_logo.jpg)
![SentenceAx Bayesian Network](pics/sentence-ax-bnet.jpg)

SentenceAx is a complete rewrite, from stem to stern, of Openie6.

https://github.com/dair-iitd/openie6

SentenceAx decomposes a compound or complex sentence
into a set of simple sentences (extractions). It does this using 
a fine-tuned BERT model.

SentenceAx is a stand-alone app, but, just like the
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

## Documentation

* [My blog posts about SentenceAx](https://qbnets.wordpress.com/?s=SentenceAx)

* Chapter entitled ``[Sentence Splitting with SentenceAx](https://github.com/rrtucci/Bayesuvius/raw/master/sentence-ax-chapter.pdf)" in my free open 
  source
book Bayesuvius 
* [Appendix](https://github.com/rrtucci/SentenceAx/raw/master/documentation/sentence-ax-appendix.pdf) to the Chapter in Bayesuvius