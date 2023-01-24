from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import argparse
import numpy as np
import os
import pandas as pd




def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
 
    x, y = [], []
  
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, min_t+20):
       
     
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
    
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x,y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    # x_offsets = np.sort( np.concatenate((np.arange(-11, 1, 1),)))
    x_offsets = np.sort( np.concatenate((np.arange(-11, 1, 1),)))
   
    y_offsets = np.sort(np.arange(1, 2, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y= generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.8)
    

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # test
    x_test, y_test= x[-num_test:], y[-num_test:]

    for cat in ["train", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y
           
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/METR-LA/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
