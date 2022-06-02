#!/usr/bin/env bash




ROOT_DIR="$HOME/venvs/sorts/SORTS"
SORTS_DIR="$ROOT_DIR/sorts"
CONTROLLERS_DIR="$SORTS_DIR/radar/controllers/src"

gcc -std=c11 -Wall -Wextra -pedantic -c -fPIC "$CONTROLLERS_DIR/controllers.c" -o "$CONTROLLERS_DIR/controllers.o"
gcc -shared "$CONTROLLERS_DIR/controllers.o" -o "$CONTROLLERS_DIR/controllers.so"
