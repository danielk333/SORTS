
Installing propagators
==============================

Orekit
-----------

Firstly check openJDK version:

.. code-block:: bash

   java -version

if OpenJDK not installed:

.. code-block:: bash

   sudo apt-get install openjdk-7-jdk

or

.. code-block:: bash

   sudo apt-get install openjdk-8-jdk

Then make sure jcc is installed:

.. code-block:: bash

   sudo apt-get install jcc

Then create a Python-2.7 environment in an appropriate folder:

.. code-block:: bash

   virtualenv env

Activate the environment:

.. code-block:: bash

   source env/bin/activate

Depending on your installation, make sure that the :code:`JCC_JDK` variable is set:

.. code-block:: bash

   export JCC_JDK="/usr/lib/jvm/java-8-openjdk-amd64"

Again, this DOES NOT work with java-9, needs 8 or 7.

Then install JCC into the environment:

.. code-block:: bash

   pip install jcc

go to: `Hipparchus <https://www.hipparchus.org/downloads.html>`_ and download binary for version 1.3.
Extract the .jar files with some archive manager, e.g. *tar*.

Clone the modified orekit including python package java classes: `Orekit with python <https://github.com/petrushy/Orekit.git>`_ .

Follow the instructions in:
`Build orekit <https://github.com/petrushy/Orekit/blob/develop/BUILDING.txt>`_

Tested building on Ubuntu 16.04:

.. code-block:: bash

   sudo apt install maven
  mvn package

If you have problem with some tests failing when building orekit, make sure you check the *petrushy/Orekit.git* 
repository status and ensure that you have the correct branch checked out before compiling (as of writing, tested branch on Ubuntu 16.04 is *develop*).

After compilation is complete, go to "/Orekit/target/" and to find the **orekit-x.jar**

Clone the python wrapper repository: `Orekit python wrapper <https://gitlab.orekit.org/orekit-labs/python-wrapper.git>`_ 

Copy the contents of the "python_files" folder (from the python wrapper repository) to the folder where you intend to build the python library.

Then place all the **hipparchus-Y.jar** files and your modified compiled **orekit-x.jar** file in your build folder.

More specifically these files are needed:

 *  orekit-x.jar
 *  hipparchus-core-1.3.jar
 *  hipparchus-filtering-1.3.jar
 *  hipparchus-fitting-1.3.jar
 *  hipparchus-geometry-1.3.jar
 *  hipparchus-ode-1.3.jar
 *  hipparchus-optim-1.3.jar
 *  hipparchus-stat-1.3.jar

A summation of these commands are

.. code-block:: bash

    wget https://www.hipparchus.org/downloads/hipparchus-1.3-bin.zip
    unzip hipparchus-1.3-bin.zip

    git clone https://github.com/petrushy/Orekit.git

    cd Orekit
    git checkout develop
    export _JAVA_OPTIONS="-Dorekit.data.path=/the/path/to/Orekit/"
    mvn package

    cd ..
    mkdir build

    git clone https://gitlab.orekit.org/orekit-labs/python-wrapper.git

    cp -v Orekit/target/orekit*.jar build/
    cp -v hipparchus-1.3-bin/*.jar build/
    cp -rv python-wrapper/python_files/* build/


Set the environment variable for building:

.. code-block:: bash

   export SRC_DIR="my/orekit/build/folder"
   export _JAVA_OPTIONS="-Dorekit.data.path=/full/path/to/Orekit/"

In this folder create a build.sh file with the following contents (remember to replace the **x**'es with the correct version compiled):

.. code-block:: bash

  #!/bin/bash

  python -m jcc \
  --use_full_names \
  --python orekit \
  --version x \
  --jar $SRC_DIR/orekit-x.jar \
  --jar $SRC_DIR/hipparchus-core-1.3.jar \
  --jar $SRC_DIR/hipparchus-filtering-1.3.jar \
  --jar $SRC_DIR/hipparchus-fitting-1.3.jar \
  --jar $SRC_DIR/hipparchus-geometry-1.3.jar \
  --jar $SRC_DIR/hipparchus-ode-1.3.jar \
  --jar $SRC_DIR/hipparchus-optim-1.3.jar \
  --jar $SRC_DIR/hipparchus-stat-1.3.jar \
  --package java.io \
  --package java.util \
  --package java.text \
  --package org.orekit \
  java.io.BufferedReader \
  java.io.FileInputStream \
  java.io.FileOutputStream \
  java.io.InputStream \
  java.io.InputStreamReader \
  java.io.ObjectInputStream \
  java.io.ObjectOutputStream \
  java.io.PrintStream \
  java.io.StringReader \
  java.io.StringWriter \
  java.lang.System \
  java.text.DecimalFormat \
  java.text.DecimalFormatSymbols \
  java.util.ArrayList \
  java.util.Arrays \
  java.util.Collection \
  java.util.Collections \
  java.util.Date \
  java.util.HashMap \
  java.util.HashSet \
  java.util.List \
  java.util.Locale \
  java.util.Map \
  java.util.Set \
  java.util.TreeSet \
  --module $SRC_DIR/pyhelpers.py \
  --reserved INFINITE \
  --reserved ERROR \
  --reserved OVERFLOW \
  --reserved NO_DATA \
  --reserved NAN \
  --reserved min \
  --reserved max \
  --reserved mean \
  --reserved SNAN \
  --build \
  --install


This command is taken from the *conda-recipe* `build sh <https://gitlab.orekit.org/orekit-labs/python-wrapper/blob/master/orekit-conda-recipe/build.sh>`_ file.

Make the file executable

.. code-block:: bash

   chmod +x build.sh

Run the build file

.. code-block:: bash

   ./build.sh

This may take some time.

Check installation by

.. code-block:: bash

   pip freeze

it should output:

.. code-block:: bash

   JCC==3.4
   orekit==9.2

Then install some additional libraries 

.. code-block:: bash

   pip install scipy
   pip install matplotlib
   pip install pytest

Make sure that you test that the installation and compilation worked.
Enter into the "test" folder (should have been part of the "python_files" folder) and run:

.. code-block:: bash

   pytest


SGP4
---------

.. code-block:: bash

   pip install sgp4

