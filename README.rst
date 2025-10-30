.. image:: logo.png
   :alt: InfraPy Logo
   :align: center
   :width: 300px

A Python Toolkit for Infrared Image Processing and Analysis
===========================================================

**InfraPy** is a modular and extensible Python library for infrared imaging. It offers a clean architecture for processing, analyzing, and visualizing infrared data, making it ideal for research, engineering, and diagnostic applications.
Whether you're exploring temperature distributions, performing thermographic inspections, or conducting advanced techniques like Thermoelastic Stress Analysis (TSA), InfraPy provides the tools and flexibility to streamline your infrared imaging workflows.

Features
--------

-  **Flexible input support**: image stacks, video files, NumPy arrays, CSV, but also .sfmov and .hcc data directly from FLIR and Telops cameras 
-  **Temperature tools**: emissivity correction, radiometric-to-temperature conversion
-  **Frequency-domain analysis**: Thermoelastic Stress Analysis via FFT and lock-in correlation analysis 
-  **Visualization**: ROI monitoring, line profiles, area averages, video animation
-  **Utility tools**: windowing, unit conversion, SNR calculation, resampling, basic image processing
-  **Modular design**: clean architecture to support GUI/CLI integration and future analysis modules

Installation
------------

Install via pip **will be soon available** on PyPI as:

.. code-block:: bash

    pip install infrapy

In the meantime, install it from source:

.. code-block:: bash

    git clone https://github.com/LolloCappo/infrapy.git
    cd infrapy
    pip install -e .

Planned Extensions
------------------

- **Time-domain analysis**: Classic thermography and temperature accumulation analysis
- **Modular design**: Cleaner architecture to support GUI/CLI integration and future analysis modules

Getting Started
---------------

**Coming soon**: example notebooks in the ``examples/`` folder for:

- Loading and displaying IR image sequences
- Performing lock-in thermoelastic analysis
- Monitoring temperature in selected ROIs
- Filtering and normalizing noisy thermal data
- Frequency-domain visualization of thermal responses

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

