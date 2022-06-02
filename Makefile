libsorts: 
	$(MAKE) -C src 
	@ln -f -s src/libsorts.so .
	@if [ "$(MAKELEVEL)" -eq "0" ]; then echo "To compile the example problems, go to a subdirectory of examples/ and execute make there."; fi

.PHONY: pythoncopy
pythoncopy:
	-cp libsorts.so `python -c "import sorts; print(sorts.__libpath__)"`
	
all: libsorts pythoncopy

clean:
	$(MAKE) -C src clean
