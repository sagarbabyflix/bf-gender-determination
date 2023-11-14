To run inference: 

cd fetal_us/src
python main.py predict configs/mks/mk000.yaml --num-workers 0 --checkpoint ../experiments/mk000/checkpoints/epoch=009-vm=1.8604.ckpt --data-dir ../../images --imgfiles ../../test_files_to_label.csv --act-fn softmax --save-preds-file ../predictions/mk000/predictions.csv

`--data-dir` argument should point to directory where the image files are saved. 
`--imgfiles` argument should point to a text file containing one path to an image file per line.
The inference script will join `data-dir` and each image file in `imgfiles` to form the full filepath name, so alternatively it is possible to use `--data-dir ''` and use absolute file paths in the filed pointed to by the `--imgfiles` argument.

To train:

cd fetal_us/src
python main.py train configs/mks/mk000.yaml --num-workers 0 --benchmark --precision 16 --gpu 2

See `src/configs` for examples of config files.
See `src/etl` for scripts preparing a CSV file for training. 
See PyTorch Lightning documentation for additional command line args.
