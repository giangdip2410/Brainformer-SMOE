
## Brainformer SMOE

This repository contains the code for running the character-level **Brainformers** from our paper. 



## Requirements
- pytorch
- fastmoe: https://github.com/laekov/fastmoe
- transformer: https://github.com/huggingface/transformers
You need CUDA 11 and PyTorch 1.10.0 or above to run this code. See [this page](https://pytorch.org/get-started/previous-versions/) for installation instructions. To replicate our experimental conditions one A100 GPU is needed. 

## Download Data
The scripts for donwloading enwiki8 and text8 datset, run:
```bash
bash get_data.sh
```

## Running experiments in the paper
##### Pretraining Brainformer on enwik8: 
The scripts for training the character-level models from the paper are located in the `./experiments/` directory. For example, to train the enwik8 model, run:
```bash
bash train.sh
```

We used eight V100 GPUs, but if you'd like to run this model on GPUs with less memory you can increase the `--batch-split`  (it splits batches into smaller pieces without changing the final result).



### The `--architecture` parameter
A standard transformer with 3 layers (so 6 self-attention and feedforward sublayers) would use be trained using  `--architecture sfsfsf`. That 6 sublayer model with a sandwiching coefficient of 1 would be  `--architecture s.sfsf.f` and with a sandwiching coefficient of 2 would be  `--architecture s.s.sf.f.f`. Make sure to also set the `--nlayers` parameter to be the length of the `architecture` string divided by 2. 


## License
The code is licensed under CC-BY-NC license. See the LICENSE file for more details.

## Acknowledgements + More Information
This code is based on the code of the [Sandwitc](https://github.com/ofirpress/sandwich_transformer) and [Adaptive Span]([https://github.com/facebookresearch/adaptive-span](https://github.com/facebookresearch/adaptive-span)) model. We recommend reading the [Adaptive Span README](https://github.com/facebookresearch/adaptive-span/blob/master/README.md) for further information on this codebase. 
