# timeflux_rasr
Implementation of rASR filtering. 

## Installation
The following steps must be performed on a Anaconda prompt console, or 
alternatively, in a Windows command console that has executed the 
`C:\Anaconda3\Scripts\activate.bat` command that initializes the `PATH` so that
the `conda` command is found.


1. Checkout this repository and change to the cloned directory
   for the following steps.

    ```
    $ git clone https://github.com/bertrandlalo/timeflux_rasr
    $ cd timeflux_rasr
    ```
    
2. Create a virtual environment with all dependencies.

    ```
    $ conda env create -f environment.yaml
    ```
    
3. Activate the environment and install this package (optionally with the `-e` 
    flag).

    ```
    $ conda activate timeflux_rasr-env
    $ pip install -e .
    ```

## References 
- Blum-Jacobsen paper
- ASR EEGLAB implementation :
    - Code : https://github.com/sccn/clean_rawdata
    - Documentation : https://sccn.ucsd.edu/wiki/Artifact_Subspace_Reconstruction_(ASR)
- rASR Matlab implementation : https://github.com/s4rify/rASRMatlab
