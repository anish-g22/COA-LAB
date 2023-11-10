
We have run the following benchmarks to evaluate the performance(IPC) using different schedulers:
1. Hotspot
2. Pathfinder
3. 3D Conv
4. BFS
5. NN
6. 3MM
 
<img>![Performance](https://github.com/anish-g22/COA-LAB/assets/97083033/b1f2c340-2d5e-46df-b31f-35726528d494)

## Table Statistics..

**LRR Scheduler**

|                      | Hotspot | Pathfinder | 3D Conv | BFS | NN | 3MM |
| -------- | ------- | ------- | ------- | ------ | ------ | -------- |
| **Without warp sharing** | 2152.852 | 1892 | 38.621 | 97.86 | 159.965 | 93.1011 |
| **With warp sharing** | 2141.581 | 1872.053 | 38.591 | 99.354 | 157.95 | 93.419 |

<br>

**GTO Scheduler**

|                      | Hotspot | Pathfinder | 3D Conv | BFS | NN | 3MM |
| -------- | ------- | ------- | ------- | ------ | ------ | -------- |
| **Without warp sharing** | 2434.57 | 2085.122 | 38.711 | 101.379 | 159.565 | 93.629 |
| **With warp sharing** | 2149.066 | 1876.66 | 38.596 | 101.37 | 161.263 | 93.1584 |

<br>

**KAWS Scheduler**

|                      | Hotspot | Pathfinder | 3D Conv | BFS | NN | 3MM |
| -------- | ------- | ------- | ------- | ------ | ------ | -------- |
| **Without warp sharing** | 2420.56 | 2090.236 | 38.694 | 102.622 | 161.605 | 93.585 |
| **With warp sharing** | 2398.17 | 2084.78 | 38.675 | 103.197 | 160.847 | 93.8401 |
