# Scientific Variable Extraction Dataset

Welcome to the repository for the Scientific Variable Extraction Dataset, as featured in our paper "[Variable Extraction for Model Recovery in Scientific Literature](https://arxiv.org/pdf/2411.14569)". This repository is structured to provide both the benchmark dataset and the corresponding evaluation results for various extraction approaches.

## Benchmark Dataset

This dataset comprises paragraphs extracted from 20 research papers focused on pandemic studies. We engaged a domain expert to meticulously annotate the scientific variables present in these papers. Details about the list of papers and the annotation guidelines can be found in the `benchmark` folder.

## Evaluation Results

The evaluation results for the different approaches tested—including conventional machine learning techniques, large language model-based solutions, and their integrations ([Palimpzest](https://palimpzest.org/research/)) — are detailed in the `evaluation` folder. These results are formatted as JSON files for ease of use and integration into further analysis.

We hope this dataset serves as a valuable resource for researchers and practitioners working on the extraction of scientific variables and enhances the understanding and application of machine learning in the domain of scientific research.

## Citation

If you use this dataset in your research, please cite it as follows:

```
@misc{liu2024variableextractionmodelrecovery,
       title={Variable Extraction for Model Recovery in Scientific Literature}, 
       author={Chunwei Liu and Enrique Noriega-Atala and Adarsh Pyarelal and Clayton T Morrison and Mike Cafarella},
       year={2024},
       eprint={2411.14569},
       archivePrefix={arXiv},
       primaryClass={cs.IR},
       url={https://arxiv.org/abs/2411.14569}, 
 }
```

Thank you for your interest in our work, and we look forward to seeing how it contributes to your research endeavors.
