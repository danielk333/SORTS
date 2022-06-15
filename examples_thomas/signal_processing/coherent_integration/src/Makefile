include Makefile.defs

OPT+= -fPIC -DLIBCOH

SOURCES=helper.cpp
OBJECTS=$(SOURCES:.c=.o)
HEADERS=$(SOURCES:.c=.h)

all: $(SOURCES) libcoh.so

%.o: %.cpp $(HEADERS)
	@echo "Compiling source file $< ..."
	$(CC) -c $(OPT) $(PREDEF) -o $@ $<

libcoh.so: $(OBJECTS)
	@echo "Linking share library $@ ..."
	$(CC) $(OPT) -shared $(OBJECTS) $(LIB) -o $@
	@echo "The shared library $< has been created successfully."

clean:
	@-rm -f *.o
	@-rm -f *.so
	@-rm -f *.gcda
	@-rm -f *.gcno
