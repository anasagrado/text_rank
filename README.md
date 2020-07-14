# text_rank

Implementation of the text rank algorithm

## Modules

The text rank algorithm is composed of two steps:
 1. Create the graph representation. In this case the nodes will be the words and the edges the number of times the word co-occur
 2. Apply the textRank algorithm to the graph representation

I based this implementation in the paper : *TextRank: Bringing Order into Texts, Rada Mihalcea and Paul Tarau*

## Examples of usage

For creating the graph represrentation first you need to provide a path with the documents that you want to calculate the co-occurrence. Example from the src directory
```python
from  create_graph import createGraph

window_size = 5
glob_path = '../data/*/C-41*'
graph_inst = createGraph(glob_path, window_size)
G = graph_inst.create_graph()
```
Second step is to run the text rank algorithm to get the keywords of the document

```python
from text_rank_algo import textRankAlgorithm

num_keywords = 5
keywords_list = textRankAlgorithm(window_size, num_keywords=num_keywords).get_keyw
```
If launching from the main you can provide different options and the result will be saved in a file


```bash
 python main.py --window_size  5 --glob_path  '../data/*/C-41*' --num_keywords 5 --output_folder "./"

```
