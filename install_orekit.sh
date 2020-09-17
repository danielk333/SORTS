#!/bin/bash

base_url="https://anaconda.org/conda-forge/orekit/10.2/download"

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
        url="$base_url/linux-64/orekit-10.2-py36h831f99a_2.tar.bz2"
    fi

    if [[ "$python_v" = "3.7" ]]; then
        url="$base_url/linux-64/orekit-10.2-py37h3340039_2.tar.bz2"
    fi

    if [[ "$python_v" = "3.8" ]]; then
        url="$base_url/linux-64/orekit-10.2-py38h950e882_2.tar.bz2"
    fi
fi

if [[ "$UNAME_S" = "Darwin" ]]; then

    if [[ "$python_v" = "3.6" ]]; then
        url="$base_url/osx-64/orekit-10.2-py36h27176af_2.tar.bz2"
    fi

    if [[ "$python_v" = "3.7" ]]; then
        url="$base_url/osx-64/orekit-10.2-py37hdadc0f0_2.tar.bz2"
    fi

    if [[ "$python_v" = "3.8" ]]; then
        url="$base_url/osx-64/orekit-10.2-py38h11c0d25_2.tar.bz2"
    fi
fi

echo "Attempting to curl from:"
echo "$url"
curl -L $url --output /tmp/orekit-10.2.tar.bz2
mkdir /tmp/orekit-10.2
tar -xjf /tmp/orekit-10.2.tar.bz2 -C /tmp/orekit-10.2/
site=$(python -c "import sys; print([x for x in sys.path if x.endswith('site-packages')][0])")

echo "Installing Orekit into $site"

cp -r /tmp/orekit-10.2/lib/python3.7/site-packages/* "$site"
echo "Installation complete"
echo $(pip freeze | grep orekit)

echo "Cleaning files"
rm -r /tmp/orekit-10.2/
rm -v /tmp/orekit-10.2.tar.bz2