# PSO-KMeans Algorithm Based on MapReduce

This repository contains the code implementation of the algorithm presented in the following paper:

**Title:** A Multi-threaded Particle Swarm Optimization-KMeans Algorithm Based on MapReduce  
**DOI:** [10.1007/s10586-024-04456-w](https://doi.org/10.1007/s10586-024-04456-w)
**Authors:** Xikang Wang, Tongxi Wang, Hua Xiang
**Developer:** Xikang Wang

## Overview
The code implements a multi-threaded particle swarm optimization (PSO) algorithm combined with KMeans clustering, designed to work efficiently in a Hadoop-based MapReduce environment.

## Experiment Environment
The environment of the experiments, including the setup and parameters, can be found in the paper. The code assumes the following:
- Hadoop (HDFS) is used for distributed storage.
- The algorithm runs in a MapReduce framework.

## Prerequisites
To run the code, you will need:
- **Hadoop**: Version 3.x or above
- **Java**: Version 8 or above
