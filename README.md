# iHOMER

**"Online Hierarchical Partitioning of the Output Space in Extreme Multi-Label Data Streams"**  
Accepted at the *European Conference on Artificial Intelligence (ECAI 2025)*.


[Paper: arxiv.org/abs/2507.20894 ](http://arxiv.org/abs/2507.20894) 


[Youtube video overview here](https://shorturl.at/IOHbW)

<img width="738" height="536" alt="Screenshot 2025-07-12 at 00 51 31" src="https://github.com/user-attachments/assets/2573f8ac-81c3-44c9-84d1-b24d256257a7" />

**iHOMER** (Incremental Hierarchy Of Multi-label ClassifiERs) is an online multi-label learning framework designed for streaming environments. It incrementally partitions the label space into disjoint, correlated clusters and dynamically adapts to concept drift, balancing scalability and predictive performance.

---

##  Repository Structure
├── code:          Core implementation of iHOMER (includes ODAC2, an adaptation of ODAC from River)


├── figures:      Visualizations, architecture, and critical difference diagrams


├── results:      Evaluation metrics and experiment logs

---

## Evaluation

We evaluate iHOMER on **23 real-world multi-label datasets** commonly used in streaming benchmarks. To reproduce the experiments, download the datasets or generate synthetic streams as outlined in the `code/` directory.

### Performance Highlights

- **+23%** over global baselines (MLHAT, MLHT, iSOUPT)  
- **+32%** over local BR-based baselines (kNN, EFDT, ARF)  
- **+40%** over random clustering strategies  

> Implementation of MLHAT adapted from: [mlhat GitHub repository](https://github.com/aestebant/mlhat)

---

## Datasets

All datasets are publicly available via the [Multi-Label Classification Dataset Repository](https://www.uco.es/kdis/mllresources/).

---

## Citation

If you use this work in your research, please cite:


@article{neves2025online,
  title={Online hierarchical partitioning of the output space in extreme multi-label data stream},
  author={Neves, Lara and Louren{\c{c}}o, Afonso and Cano, Alberto and Marreiros, Goreti},
  journal={arXiv preprint arXiv:2507.20894},
  year={2025}
}
