.. image:: logo.png
   :alt: InfraPy Logo
   :align: center
   :width: 300px

Infrared Image Processing in Python
=============================================

**InfraPy** is a modular and extensible Python library for infrared image analysis. It offers a clean architecture for processing, analyzing, and visualizing infrared data, making it ideal for research, engineering, and diagnostic applications.
Whether you're exploring temperature distributions, performing thermographic inspections, or conducting advanced techniques like Thermoelastic Stress Analysis (TSA), InfraPy provides the tools and flexibility to streamline your infrared imaging workflows.

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

Install via pip:

.. code-block:: bash

    pip install infrapy

Or install from source:

.. code-block:: bash

    git clone https://github.com/LolloCappo/infrapy.git
    cd infrapy
    pip install -e .

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

