:github_url: https://github.com/merlinquantum/merlin

===========
Performance
===========

MerLin performance notes collect measured CPU and GPU behavior for quantum
layers and reproduced model paths. Use these pages to estimate hardware
requirements before scaling a benchmark.

.. toctree::
   :maxdepth: 1
   :hidden:

   QuantumLayer GPU performance <performance>
   Photonic QGAN GPU performance <performance_gpu>

Overall Performance
-------------------

.. merlin-gallery::
   :data: _data/galleries/performance/overall.json
   :columns: 2
   :contour-color: #5648ED

Models Performance
------------------

.. merlin-gallery::
   :data: _data/galleries/performance/models.json
   :columns: 2
   :contour-color: #5648ED

Scaling study of QCNN
=========================
.. toctree::
   :hidden:
   :maxdepth: 1

   scaling_study_of_qcnn
