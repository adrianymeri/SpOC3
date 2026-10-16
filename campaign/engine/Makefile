# Build the shared engine. -march=native uses the build machine's vector
# units (AVX-512 on the Xeon Gold 6130 server); build it on the machine
# that runs the campaign (campaign/install.sh does).
CC ?= cc
CFLAGS ?= -O3 -march=native -fPIC -Wall -Wextra -Wno-unused-parameter
libtorso.so: torso_engine.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lm
clean:
	rm -f libtorso.so
