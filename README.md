# Learning to Play Sequential Games versus Unknown Opponents


This repository contains the code associated to the paper:
> [**Learning to Play Sequential Games versus Unknown Opponents**](https://arxiv.org/pdf/2007.05271.pdf)
> *Pier Giuseppe Sessa, Ilija Bogunovic, Maryam Kamgarpour, Andreas Krause*.
> Neural Information Processing Systems (NeurIPS), 2020.

Usage
-- 

You can install the required dependences via: 
```setup
pip install -r requirements.txt
```

- The folder **Traffic_routing/** contains the code associated to the Traffic Routing experiment of Section 4.1.
The script `Main_Script.py` reproduces the results of the paper, namely comparing the considered routing strategies with the proposed StackelUCB algorithm.
Results can be plotted running the script `Plotting.py`

- The folder **Wildlife_protection/** concerns the Wildlife conservation task of Section 4.2. 
The script `Main_Script.py` setups the considered experiment and compares the discussed strategies for the rangers.
Results can be plotted running the script `Plotting.py`

