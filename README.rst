.. image:: logo.png
   :alt: InfraPy Logo
   :align: center
   :width: 300px

Infrared Image Processing in Python
=============================================
**InfraPy** is a modular Python library for infrared image analysis. It provides a clean and extensible structure for processing, analyzing, and visualizing infrared data.

The library is designed to serve research, engineering, and diagnostic applications where infrared imaging is key — from temperature field exploration to dynamic lock-in analysis.

Features
--------

-  **Flexible input support**: image stacks, video files, NumPy arrays, CSV
-  **Temperature tools**: emissivity correction, radiometric-to-temperature conversion
-  **Time-domain analysis**: Thermography, temperature accumulation analysis
-  **Frequency-domain analysis**: Thermoelastic Stress Analysis
-  **Visualization**: ROI monitoring, line profiles, area averages, video animation
-  **Utility tools**: windowing, unit conversion, SNR calculation, resampling, image processing
-  **Modular design**: clean architecture to support GUI/CLI integration and future analysis modules

Installation
------------

Install via pip (once published):

.. code-block:: bash

    pip install infrapy

Or install from source:

.. code-block:: bash

    git clone https://github.com/LolloCappo/infrapy.git
    cd infrapy
    pip install -e .

Library Structure
-----------------

.. code-block:: text

    infrapy/
    ├── io.py                   # Load IR data from files and save results
    ├── emissivity.py           # Emissivity-based analysis
    ├── visualization.py        # Visualization tools
    ├── thermography.py         # Thermography-based analysis
    ├── thermoelasticity.py     # Thermoelasticity-based analysis 
    └── utils.py                # General-purpose utilities


Planned Extensions
------------------

- Thermal anomaly detection tools
- GUI frontend for interactive workflows

Getting Started
---------------

Coming soon: example notebooks in the ``examples/`` folder for:

- Loading and displaying IR image sequences
- Performing lock-in thermoelastic analysis
- Monitoring temperature in selected ROIs
- Filtering and normalizing noisy thermal data

Contributing
------------

Feel free to contribute! Open issues for bug reports, feature suggestions, or development help. Pull requests are welcome.

License
-------

MIT License

Contact
-------

Project Lead: Lorenzo Capponi

Email: lorenzocapponi@outlook.it

GitHub: https://github.com/LolloCappo/infrapy


Acknowledgements
-------------------
**InfraPy** was developed within the framework of the `ARTEMIDE`_ project, funded by the European Research Agency (ERA) under the Maria Skłodowska Curie Actions (MSCA). Grant id is 101180595.


.. _ARTEMIDE: http://ladisk.si/?what=incfl&flnm=artemide.php

