# Graph-based skeleton data compression

With the advancement of reliable, fast, portable acquisition systems, human motion capture data is becoming widely used in many industrial, medical, and surveillance applications. These systems can track multiple people simultaneously, providing full-body skeletal keypoints as well as more detailed landmarks in face, hands and feet. This leads to a huge amount of skeleton data to be transmitted or stored. In this paper, we introduce Graph-based Skeleton Compression (GSC), an efficient graph-based method for nearly lossless compression. We use a separable spatio-temporal graph transform along with non-uniform quantization, coefficient scanning and entropy coding with run-length codes for nearly lossless compression. We evaluate the compression performance of the proposed method on the large NTU-RGB activity dataset. Our method outperforms a 1D discrete cosine transform method applied along temporal direction. In near-lossless mode our proposed compression does not affect action recognition performance.

## Install, Build, Run
```bash
git clone --recursive https://github.com/nguyen-huong/skel_compression.git
```
