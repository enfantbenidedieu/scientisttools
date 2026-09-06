.. _install:

=======
Install
=======

.. tip::
    This page assumes you are comfortable using a terminal and are familiar with package managers. The only prerequisite for installing scientisttools is Python itself.

Virtual environment
~~~~~~~~~~~~~~~~~~~

Install the 64-bit version of Python 3, for instance from the `official website <https://www.python.org/>`_. Now create a `virtual environment (venv) <https://docs.python.org/3/tutorial/venv.html>`_ and install scientisttools.

.. note::
    The virtual environment is optional but strongly recommended, in order to avoid potential conflicts with other packages.

.. code-block:: console

    PS C:\> python -m venv scientisttools-env # create virtual env
    PS C:\> scientisttools-env\Scripts\activate  # activate
    PS C:\> pip install -U scientisttools  # install scientisttools

Version
~~~~~~~

In order to check your installation, you can use.

.. code:: python

    >>> import scientisttools
    >>> print(scientisttools.__version__)
    0.2.0post1

Using an isolated environment such as pip venv or conda makes it possible to install a specific version of scientisttools with pip and conda and its dependencies independently of any previously installed Python packages.

.. note::
    You should always remember to activate the environment of your choice prior to running any Python command whenever you start a new terminal session.

Dependencies
~~~~~~~~~~~~

scientisttools is compatible with python version which supports both dependencies :

==============  =========
Dependency      Version
==============  =========
numpy           1.21
pandas          1.4
scikit-learn    1.2
statsmodels     0.14.6
plotnine        0.10.1
openpyxl        3.1.5
adjustText      0.8.2
pyreadr         0.5.4
mizani          0.14.4
tabulate        0.9.0
==============  =========