#!/bin/bash

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
curl -L $url --output /tmp/orekit-10.3.tar.bz2
mkdir /tmp/orekit-10.3
tar -xjf /tmp/orekit-10.3.tar.bz2 -C /tmp/orekit-10.3/
site=$(python -c "import sys; print([x for x in sys.path if x.endswith('site-packages')][0])")
pac_source="/tmp/orekit-10.3/lib/python$python_v/site-packages"

echo "Installing Orekit into $site"
echo "From $pac_source"

cp -r "$pac_source"/* "$site"
echo "Installation complete"
echo $(pip freeze | grep orekit)

echo "Cleaning files"
rm -r /tmp/orekit-10.3/
rm -v /tmp/orekit-10.3.tar.bz2