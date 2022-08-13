CC=gcc
CFLAGS=-fPIC
LIBS=-lm

SOURCES=radar_controller.c static_priority_scheduler.c plotting_controls.c signals.c measurements.c radar.c
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
