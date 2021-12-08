#!/usr/bin/env bash

# 
# This installation method needs the following to be installed before
# - openjdk-8
# - maven
# - JCC (pip install)
# 

if [ -z "$2" ]; then
    TARGET_DIR="$HOME/tmp/orekit_build"
    echo "No custom build path given, using $TARGET_DIR"
else
    TARGET_DIR=$2
fi

JDK_LOC=$(find /usr/lib/jvm/ -name "*8-openjdk*")

case "$1" in
check)
    echo "Checking java"
    java -version
    if [ $? -eq 0 ]; then
        echo "[JAVA OK]"
    else
        echo "[JAVA FAIL]"
    fi

    echo "Checking JCC"
    CH_TEST=$(pip freeze | grep JCC)
    echo "$CH_TEST"
    if [ -z "$CH_TEST" ]; then
        echo "[JCC FAIL]"
    else
        echo "[JCC OK]"
    fi

    echo "Checking 8-openjdk"
    echo $JDK_LOC
    if [ -z "$JDK_LOC" ]; then
        echo "[JDK FAIL]"
    else
        echo "[JDK OK]"
    fi
    
    echo "Checking maven"
    mvn --version
    if [ $? -eq 0 ]; then
        echo "[MAVEN OK]"
    else
        echo "[MAVEN FAIL]"
    fi
    ;;
build)
    mkdir -p $TARGET_DIR
    mkdir -p $TARGET_DIR/build
    cd $TARGET_DIR

    git clone -b python-wrapper-additions-V10.3.1 https://github.com/petrushy/Orekit.git 
    git clone https://gitlab.orekit.org/orekit-labs/python-wrapper.git
    wget https://packages.orekit.org/repository/maven-releases/org/orekit/rugged/2.2/rugged-2.2.jar
    wget https://www.hipparchus.org/downloads/hipparchus-1.8-bin.tar.bz2

    mv rugged-2.2.jar ./build/ -v

    tar xjf hipparchus-1.8-bin.tar.bz2
    cp hipparchus-1.8-bin/*.jar ./build/ -v

    cd Orekit
    mvn package

    cp ./target/*.jar ../build/ -v

    cd ..
    cp -rv python-wrapper/python_files/* build/

    ;;
install)

    cd $TARGET_DIR/build

    SRC_DIR="$TARGET_DIR/build"

    if [ "$(uname)" == "Darwin" ]
    then
      export JCC_JDK=${JDK_LOC}
      export JCC_ARGSEP=";"
      export JCC_INCLUDES="${JDK_LOC}/include;${JDK_LOC}/include/darwin"
      export JCC_LFLAGS="-v;-L${JDK_LOC}/jre/lib;-ljava;-L${JDK_LOC}/jre/lib/server;-ljvm;-Wl,-rpath;-Wl,${JDK_LOC}/jre/lib;-Wl,-rpath;-Wl,${JDK_LOC}/jre/lib/server;-mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
        export JCC_CFLAGS="-fno-strict-aliasing;-Wno-write-strings;-Qunused-arguments;-mmacosx-version-min=10.9;-std=c++11;-stdlib=libc++"
      export JCC_DEBUG_CFLAGS="-O0;-g;-DDEBUG"
      export JCC_JAVAC="javac;-source;1.8;-target;1.8"
      export JCC_JAVADOC="javadoc"

    else
      # GNU/Linux recipe
      export JCC_JDK=${JDK_LOC}
      export JCC_ARGSEP=";"
        export JCC_LFLAGS="-v;-Wl,-v;-L${JDK_LOC}/jre/lib/amd64;-ljava;-L${JDK_LOC}/jre/lib/amd64/server;-ljvm;-lverify;-Wl,-rpath=${JDK_LOC}/jre/lib/amd64:${JDK_LOC}/jre/lib/amd64/server"
      export JCC_INCLUDES="${JDK_LOC}/include;${JDK_LOC}/include/linux"
        export JCC_JAVAC=${JDK_LOC}/bin/javac
        export JCC_CFLAGS="-v;-fno-strict-aliasing;-Wno-write-strings;-D__STDC_FORMAT_MACROS"
      export JCC_DEBUG_CFLAGS="-O0;-g;-DDEBUG"
      export JCC_JAVADOC="javadoc"
    fi

    printenv

    python -m jcc \
    --use_full_names \
    --python orekit \
    --version 10.3.1 \
    --jar $SRC_DIR/orekit-10.3.1.jar \
    --jar $SRC_DIR/hipparchus-clustering-1.8.jar \
    --jar $SRC_DIR/hipparchus-core-1.8.jar \
    --jar $SRC_DIR/hipparchus-fft-1.8.jar \
    --jar $SRC_DIR/hipparchus-filtering-1.8.jar \
    --jar $SRC_DIR/hipparchus-fitting-1.8.jar \
    --jar $SRC_DIR/hipparchus-geometry-1.8.jar \
    --jar $SRC_DIR/hipparchus-migration-1.8.jar \
    --jar $SRC_DIR/hipparchus-ode-1.8.jar \
    --jar $SRC_DIR/hipparchus-optim-1.8.jar \
    --jar $SRC_DIR/hipparchus-stat-1.8.jar \
    --jar $SRC_DIR/rugged-2.2.jar \
    --package java.io \
    --package java.util \
    --package java.text \
    --package org.orekit \
    --package org.orekit.rugged \
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
    java.util.ArrayDeque  \
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
    java.util.stream.Collectors \
    java.util.stream.Stream \
    java.util.stream.DoubleStream \
    java.util.function.LongConsumer \
    java.util.function.IntConsumer \
    java.util.function.DoubleConsumer \
    --module ./pyhelpers.py \
    --reserved INFINITE \
    --reserved ERROR \
    --reserved OVERFLOW \
    --reserved NO_DATA \
    --reserved NAN \
    --reserved min \
    --reserved max \
    --reserved mean \
    --reserved SNAN \
    --classpath $JDK_LOC/lib/tools.jar \
    --files 81 \
    --build \
    --install
    
    ;;
clean) 
    rm -rv Orekit
    rm -v hipparchus-1.8-bin.tar.bz2
    rm -rv hipparchus-1.8-bin
    rm -rv python-wrapper
    rm -rv build
    ;;
lazy)
    base_url="https://anaconda.org/conda-forge/orekit/10.3/download"

    python_v_min=3.6
    python_v=$(python -c 'import sys; print("%d.%d"% sys.version_info[0:2])' )
    python_check=$(python -c "import sys; print(int(float('%d.%d'% sys.version_info[0:2]) >= $python_v_min))")

    if [[ "$python_check" = "0" ]]; then
        error "Python version is $python_v but must be >= $python_v_min"
    else
        echo "Getting Orekit for Python $python_v"
    fi

    UNAME_S=$(uname -s)

    if [[ "$UNAME_S" = "Linux" ]]; then

        if [[ "$python_v" = "3.6" ]]; then
            url="$base_url/linux-64/orekit-10.3-py36hc4f0c31_3.tar.bz2"
        fi

        if [[ "$python_v" = "3.7" ]]; then
            url="$base_url/linux-64/orekit-10.3-py37hcd2ae1e_3.tar.bz2"
        fi

        if [[ "$python_v" = "3.8" ]]; then
            url="$base_url/linux-64/orekit-10.3-py38h709712a_3.tar.bz2"
        fi

        if [[ "$python_v" = "3.9" ]]; then
            url="$base_url/linux-64/orekit-10.3-py39he80948d_3.tar.bz2"
        fi
    fi

    if [[ "$UNAME_S" = "Darwin" ]]; then

        if [[ "$python_v" = "3.6" ]]; then
            url="$base_url/osx-64/orekit-10.3-py36hefe7e0e_3.tar.bz2"
        fi

        if [[ "$python_v" = "3.7" ]]; then
            url="$base_url/osx-64/orekit-10.3-py37hd8d24ac_3.tar.bz2"
        fi

        if [[ "$python_v" = "3.8" ]]; then
            url="$base_url/osx-64/orekit-10.3-py38ha048514_3.tar.bz2"
        fi

        if [[ "$python_v" = "3.9" ]]; then
            url="$base_url/osx-64/orekit-10.3-py39h9fcab8e_3.tar.bz2"
        fi
    fi

    echo "Attempting to curl from:"
    echo "$url"
    curl -L $url --output $TARGET_DIR/orekit-10.3.tar.bz2
    mkdir $TARGET_DIR/orekit-10.3
    tar -xjf $TARGET_DIR/orekit-10.3.tar.bz2 -C $TARGET_DIR/orekit-10.3/
    site=$(python -c "import sys; print([x for x in sys.path if x.endswith('site-packages')][0])")
    pac_source="$TARGET_DIR/orekit-10.3/lib/python$python_v/site-packages"

    echo "Installing Orekit into $site"
    echo "From $pac_source"

    cp -r "$pac_source"/* "$site"
    echo "Installation complete"
    echo $(pip freeze | grep orekit)

    echo "Cleaning files"
    rm -r $TARGET_DIR/orekit-10.3/
    rm -v $TARGET_DIR/orekit-10.3.tar.bz2
    ;;
*)
    echo "Command not found, exiting"
    ;;
esac


