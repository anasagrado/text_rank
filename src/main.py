import argparse
from glob import glob
import os
from create_graph import createGraph
from text_rank_algo import textRankAlgorithm


def make_parser():
    parser = argparse.ArgumentParser(description='Calculate the occurrence graph.')
    parser.add_argument(
        '--window_size',
        default=5, type=int,
        help='window size for the occurrence (default: 2)'
    )
    parser.add_argument(
        '--glob_path',
        default="./*.txt",
        help='Unix like path where the documents are stored.' +
                'Default will look for txt files in the current directory. (default: ./*.txt)'
    )
    parser.add_argument(
        '--num_keywords',
        default= 5, type=int,
        help='Number of keywords to write (default: 5)'
    )
    parser.add_argument(
        '--output_folder',
        default="./",
        help='Folder to create the files with the output  (default: current directory)'
    )

    return parser

if __name__ == "__main__":

    parser = make_parser()
    input_args = parser.parse_args()

    glob_path = input_args.glob_path
    window_size = input_args.window_size
    num_keywords = input_args.num_keywords
    output_folder = input_args.output_folder

    print("Global path to read the data from {}".format(glob_path))
    print("Window size used to construct the graph {}".format(window_size))
    print("Number of keywords to output {}".format(num_keywords))
    print("Output folder name will be {}".format(output_folder))

    for file in glob(glob_path):
        file_name = os.path.basename(file)
        path_to_save = output_folder + file_name + ".result"

        graph_inst = createGraph(file , window_size)
        G = graph_inst.create_graph()

        keywords_list = textRankAlgorithm(window_size, num_keywords=num_keywords).get_keywords(G)
        with open( path_to_save , mode='wt', encoding='utf-8') as f:
            f.write('\n'.join(keywords_list))
