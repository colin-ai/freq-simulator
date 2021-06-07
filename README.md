# sysiPy

sysiPy (Synthetic Simulation Python) is a Python library aims to easily generate simple 
synthetic dataset for fast prototyping in machine learning area.

## Installation
```git clone```


## How It Works
The aim of library is to generate a dataset ready-to-use with a minimum workflow.
The process folows 2 main steps :
- A common task to any model dedicated to generate pseudo-random sequences of inputs.
- From these inputs, a specific task which computes a sequence of outpus according 
  to a model. 

## Models available
From simpliest to the least simple model.

### Tracking velocity



### Tracking signal's frequency
For each sequence three quantities linked to a signal are given :  
- Frequency.
- Derivative of frequency.
- Phase.

### 1D-tire slip from the Pacejka Magic Formula
It simulates a 1D wheel tire slip and. For each sequence two quantities are given :  
- True velocity, meaning without slippage. 
- Wheel velocity, which includes slippage.

## Licence
MIT

## Remarks
This code has a research purpose.