# MANA：Exploiting Associations among Multi-Aspect Node Properties in Heterogeneous Graphs for Link Prediction



- Amazon

  ```bash
  python main_link_simple.py --dataset amazon --model SimpleLinkDeep --hidden 256 --embed-size 256 --epochs 200 --batch-size 8192 --num-hops 3 --bns --lr 0.001
  ```

- LastFM

  ```bash
  python main_link_simple.py --dataset LastFM --model SimpleLinkDeep --hidden 256 --embed-size 256 --epochs 200 --batch-size 8192 --num-hops 3 --bns --lr 0.001
  ```

- PubMed

  ```bash
  python main_link_simple.py --dataset PubMed --model SimpleLinkDeep --hidden 256 --embed-size 256 --epochs 200 --batch-size 8192 --num-hops 3 --bns --lr 0.001
  ```

  
