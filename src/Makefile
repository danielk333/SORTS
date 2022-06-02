include Makefile.defs

OPT+= -fPIC -DLIBSORTS

SOURCES=radar_controller.c
OBJECTS=$(SOURCES:.c=.o)
HEADERS=$(SOURCES:.c=.h)

all: $(SOURCES) libsorts.so

%.o: %.c $(HEADERS)
	@echo "SORTS installation -> Compiling source file $< ..."
	$(CC) -c $(OPT) $(PREDEF) -o $@ $<

libsorts.so: $(OBJECTS)
	@echo "SORTS installation -> Linking share library $@ ..."
	$(CC) $(OPT) -shared $(OBJECTS) $(LIB) -o $@
	@echo "SORTS installation -> The shared library $< has been created successfully."

clean:
	@echo "SORTS installation -> Removing object files *.o ..."
	@-rm -f *.o
	@echo "SORTS installation -> Removing shared library librebound.so ..."
	@-rm -f *.so
	@echo "SORTS installation -> Removing coverage files ..."
	@-rm -f *.gcda
	@-rm -f *.gcno
