Photonic QGAN
====================

A Generative Adversarial Network (GAN) is a classical neural network that tries to create a  model that generates new samples from a given distribution. Here we will present a photonic implementation of this model that is ready-to-use in MerLin. It was first presented in Sedrakyan and Salavrakos' `paper <https://opg.optica.org/opticaq/fulltext.cfm?uri=opticaq-2-6-458>`_. There is also a reproduction in MerLin of this paper `here <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/photonic_QGAN>`_.

Let's first understand how a classical GAN works.

Classical GAN
--------------

The model is composed of two different submodules: a generator G and a discriminator D which are competing against one another in an adversarial zero-sum game. The generator wants to fool the discriminator with fake images and the discriminator wants to correctly identify fake images versus real images from the dataset. The generator receives random noise values as input and transforms them to fake images of the dataset. The model can be identified in the next figure where we can just consider the generator as a whole classical model without the quantum sub generators.

.. image:: ../../_static/reproduced_papers/photonicQGAN.png
   :alt: Photonic QGAN model.
   :width: 1200px
   :align: center


The GAN has a specific training routine. The main steps are:

#. For a number of ``iterations``:

   #. For a number of discriminator iterations ``d_steps``:

      #. Train the discriminator parameters with batches of fake and real
         images. The labels of the real values are ``real_labels`` and the
         fake images' labels are ``fake_labels``.

         After hyperparameter optimization, we choose:

         - ``real_labels = 0.9``
         - ``fake_labels = 0.0``

   #. For a number of generator iterations ``g_steps``:

      #. Train the generator parameters using only its generated fake images,
         which are classified by the discriminator. The associated labels are
         ``generator_labels``.

         After hyperparameter optimization, we choose:

         - ``generator_labels = 0.9``

The loss calculations for both steps are:

#. **Discriminator training's loss**

   .. math::

      \mathcal{L}_D =
      \frac{1}{n}\sum_{i=1}^{n}
      \mathrm{Loss}\!\left(D(x_i), \text{real labels}\right)
      +
      \frac{1}{n}\sum_{i=1}^{n}
      \mathrm{Loss}\!\left(D(G(z_i)), \text{fake labels}\right)

   where :math:`x_i` are the real images in the batch and :math:`z_i`
   are the noise vectors generated for the batch.

#. **Generator training's loss**

   .. math::

      \mathcal{L}_G =
      \frac{1}{n}\sum_{i=1}^{n}
      \mathrm{Loss}\!\left(D(G(z_i)), \text{generator labels}\right)

   where :math:`z_i` are the noise vectors generated for the batch.

For the photonic QGAN, the preferred loss function is
``torch.nn.BCEWithLogitsLoss`` from PyTorch.

Photonic QGAN
-------------
The photonic implementation of the QGAN is similar. Just like it is illustrated in the figure above, the generator will be composed of multiple photonic circuits that will be run together to generate brand new images. The concept and training of the GAN model stays the same.

MerLin implementation
----------------------
The ready-to-use MerLin model creats the generator. It is defined as the :class:`~merlin.models.photonic_generator.PhotonicGenerator` object. This model takes noise as input and generates new features that are close to the data distribution as an output.

There is also a helper classes to help transform the output.

* ``ImageAdapter``: Adapt tensor measurements to GAN-native image tensors.
* ``VectorAdapter``: Concatenate tensor measurements into fixed-width vectors.

Although the :class:`~merlin.models.photonic_generator.PhotonicGenerator`'s output adapter argument can accept a regular torch module as well.

For more information on the specifications of these objects, please consult the api: :doc:`/api_reference/api/merlin.models.photonic_generator`.

User recommendations and limitations
------------------------------------

- When choosing the size of the circuits in the quantum layers that form the photonic generator, follow the same guidelines as for other MerLin circuits. Further information on limitations on number of photons and modes can be found `here <https://merlinquantum.ai/performance/index.html#pushing-the-h100-to-its-limits>`_.  
- As a rule of thumb, when choosing the number of generator heads for patch generaton through parameter 'count', we recommend that the total number of pixels of the image (or size of the data) should be lower or equal to the number of generator heads times the output space size of each quantum circuit.
- For images, we also recommend making sure that the patches are at least of size 2x2 pixels.

Tutorial
----------------
A tutorial on the use of the :class:`~merlin.models.photonic_generator.PhotonicGenerator` to create a photonic QGAN is available by clicking on the next window.

.. merlin-gallery::
   :data: _data/galleries/user_guide/models/qgan_notebook.json
   :columns: 2

Citation
--------

.. code-block:: bibtex

   @article{sedrakyan2025photonicqgan,
     title={Photonic quantum generative adversarial networks for classical data},
     author={Sedrakyan, Tigran and Salavrakos, Alexia},
     journal={Optica Quantum},
     volume={2},
     number={6},
     pages={458--467},
     year={2025},
     doi={10.1364/OPTICAQ.530346}
   }
