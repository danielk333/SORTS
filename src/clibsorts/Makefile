CC=gcc
CFLAGS=-fPIC
LIBS=-lm

SOURCES=$(wildcard **/*.c)
OBJECTS=$(SOURCES:.c=.o)
EXT=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTLIB=clibsorts$(EXT)

all: $(OUTLIB)

%.o: %.c
	@echo "SORTS installation -> Compiling source file $< ..."
	$(CC) -c $(CFLAGS) -o $@ $<

$(OUTLIB): $(OBJECTS)
	@echo "SORTS installation -> Linking share library $@ ..."
	$(CC) $(CFLAGS) -shared $(OBJECTS) $(LIBS) -o $@
	@mv $@ ../sorts/ -v
	@echo "SORTS installation -> The shared library $< has been created successfully."

clean:
	@echo "SORTS installation -> Removing object files *.o ..."
	@-rm -f *.o
	@echo "SORTS installation -> Removing shared library *.so ..."
	@-rm -f *.so
