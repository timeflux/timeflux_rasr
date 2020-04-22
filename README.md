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

4. (optional) If you have a problem with a missing package, add it to the `environment.yaml`, then:
    ```
    (timeflux_rasr-env)$ conda env update --file environment.yaml
    ```

5. (optional) If you want to use the notebook, we advice Jupyter Lab (already in requirements) with additional steps:
    ```
    $ conda activate timeflux_rasr-env
    # install jupyter lab 
    (timeflux_rasr-env)$ conda install -c conda-forge jupyterlab 
    (timeflux_rasr-env)$ ipython kernel install --user --name=timeflux_rasr-env 
    (timeflux_rasr-env)$ jupyter lab  # run jupyter lab and select timeflux_rasr-env kernel
    # quit jupyter lab with CTRL+C then
    (timeflux_rasr-env)$ conda install -c conda-forge ipympl
    (timeflux_rasr-env)$ conda install -c conda-forge nodejs 
    (timeflux_rasr-envv)$ jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
    ```
   
    To test if widget is working if fresh notebook:
    ```
    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib widget
    
    df = pd.DataFrame({'a': [1,2,3]})
    
    plt.figure(2)
    plt.plot(df['a'])
    plt.show()
    ```

## References 
- Blum-Jacobsen paper
- ASR EEGLAB implementation :
    - Code : https://github.com/sccn/clean_rawdata
    - Documentation : https://sccn.ucsd.edu/wiki/Artifact_Subspace_Reconstruction_(ASR)
- rASR Matlab implementation : https://github.com/s4rify/rASRMatlab
