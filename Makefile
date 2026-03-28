# Exact paths from spack
PARMETIS_DIR = /home/humayun/spack/opt/spack/linux-x86_64_v2/parmetis-4.0.3-wzrz7k2ykbkwyyn24hhh7taa36nfc6pb
METIS_DIR    = /home/humayun/spack/opt/spack/linux-x86_64_v2/metis-5.1.0-3hajfgdabym3gmyjzcsmunyfdo2ak6gg

CC      = mpicc
CFLAGS  = -O2 -Wall \
           -I$(PARMETIS_DIR)/include \
           -I$(METIS_DIR)/include
LDFLAGS = -L$(PARMETIS_DIR)/lib \
           -L$(METIS_DIR)/lib \
           -lparmetis -lmetis -lm

SRC_DIR = src

all: pagerank_p2p pagerank_collective pagerank_async

pagerank_p2p: $(SRC_DIR)/main.c $(SRC_DIR)/pagerank_p2p.c $(SRC_DIR)/graph_utils.c
	$(CC) $(CFLAGS) -DSCENARIO=1 -o $@ $^ $(LDFLAGS)

pagerank_collective: $(SRC_DIR)/main.c $(SRC_DIR)/pagerank_collective.c $(SRC_DIR)/graph_utils.c
	$(CC) $(CFLAGS) -DSCENARIO=2 -o $@ $^ $(LDFLAGS)

pagerank_async: $(SRC_DIR)/main.c $(SRC_DIR)/pagerank_async.c $(SRC_DIR)/graph_utils.c
	$(CC) $(CFLAGS) -DSCENARIO=3 -o $@ $^ $(LDFLAGS)

clean:
	rm -f pagerank_p2p pagerank_collective pagerank_async
