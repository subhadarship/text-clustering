# text-clustering
clustering text data (data mining fall 2019)

NOTE: `present.html` is created using revel.js


`bleu_score` function in `bleu.py` takes `candidate_corpus` as first argument and `reference corpus` as second argument. But the order is reversed in the `calculate_bleu` function in [6 - Attention is All You Need.ipynb](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb), [5 - Convolutional Sequence to Sequence Learning.ipynb](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb) and [4 - Packed Padded Sequences, Masking, Inference and BLEU.ipynb](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb). The updated `calculate_bleu` function is:

```
def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        trgs.append([trg])
        pred_trgs.append(pred_trg)
        
    return bleu.bleu_score(pred_trgs, trgs)

```

