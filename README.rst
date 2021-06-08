Timeflux rASR
=============

This plugin performs real-time artifact correction on EEG data.

Installation
------------

First, make sure that `Timeflux <https://github.com/timeflux/timeflux>`__ is installed.

You can then install this plugin in the ``timeflux`` environment:

::

    $ conda activate timeflux
    $ pip install timeflux_rasr


If you encounter issues with ``pyRiemann``, try installing the latest version directly from the repository:

::

    $ pip install git+https://github.com/pyRiemann/pyRiemann


Witness the magic!
------------------

::

    $ timeflux -d examples/replay.yaml

Screenshot
----------

.. image:: https://github.com/timeflux/timeflux_rasr/raw/master/doc/static/img/screenshot.png

References
----------

* Blum-Jacobsen `paper <https://www.frontiersin.org/articles/10.3389/fnhum.2019.00141/full>`__
* ASR EEGLAB implementation (`code <https://github.com/sccn/clean_rawdata>`__, `documentation <https://sccn.ucsd.edu/wiki/Artifact_Subspace_Reconstruction_(ASR)>`__)
* rASR `Matlab implementation <https://github.com/s4rify/rASRMatlab>`__
