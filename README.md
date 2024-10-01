# HGM-SRR
Hypergraph Mamba Reasoning Based Social Relation Recognition

## Environment

Please refer to the "environment.txt".

## Dataset
[PISC](https://zenodo.org/record/1059155#.WznPu_F97CI) was released by [[Li et al. ICCV 2017](https://arxiv.org/abs/1708.00634)]. It involves a two-level relationship, i.e., coarse-level relationships with 3 categories and fine-level relationships with 6 categories.

[PIPA-relation](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/human-activity-recognition/social-relation-recognition/) was released by [[Sun et al. CVPR 2017](https://arxiv.org/abs/1704.06456)]. It covers 5 social domains, which can be further divided into 16 social relationships. On this dataset, we focus on the 16 social relationships.

## Usage
    optional arguments:
      -h, --help                show this help message and exit
      -j N, --workers N         number of data loading workers (defult: 4)
      -b N, --batch-size N      mini-batch size (default: 1)
      --print-freq N, -p N      print frequency (default: 10)
      --weights PATH            path to weights (default: none)
      --scale-size SCALE_SIZE   input size
      --world-size WORLD_SIZE   number of distributed processes
      -n N, --num-classes N     number of classes / categories
      --write-out               write scores
      --adjacency-matrix PATH   path to adjacency-matrix of graph
      --crop-size CROP_SIZE     crop size
      --result-path PATH        path for saving result (default: none)

## Contributing
For any questions, feel free to open an issue or contact us (tangwang@stu.scu.edu.cn)
