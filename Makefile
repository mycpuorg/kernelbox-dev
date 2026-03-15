CC        = gcc
CUDA_HOME ?= $(shell nvcc_path=$$(which nvcc 2>/dev/null) && dirname $$(dirname "$$nvcc_path") || echo /usr/local/cuda)

CFLAGS    = -Wall -Wextra -O2 -g -I$(CUDA_HOME)/include -Isrc -MMD -MP
LDFLAGS   = -lcuda -lnvrtc -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs

B         = build
OBJ       = $(B)/obj
TEST_BIN  = $(B)/tests
TOOL_BIN  = $(B)/tools

CORE_SRCS = src/cuda_check.c src/ipc.c src/vmm.c
CORE_OBJS = $(patsubst src/%.c,$(OBJ)/%.o,$(CORE_SRCS))
DEPS      = $(CORE_OBJS:.o=.d)

C_TESTS   = $(TEST_BIN)/test_cuda_check $(TEST_BIN)/test_ipc_fd
TOOLS     = $(TOOL_BIN)/kbox $(TOOL_BIN)/kbox_worker_daemon

PREFIX    ?= /usr/local
BINDIR    = $(PREFIX)/bin
LIBEXECDIR = $(PREFIX)/libexec/kbox

.PHONY: all tools test test-cli test-all clean install uninstall sync-python compile_commands.json

all: $(CORE_OBJS) $(C_TESTS) $(TOOLS)

tools: $(TOOLS)

$(OBJ) $(TEST_BIN) $(TOOL_BIN):
	@mkdir -p $@

$(OBJ)/%.o: src/%.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_BIN)/test_cuda_check: tests/test_cuda_check.c $(CORE_OBJS) | $(TEST_BIN)
	$(CC) $(CFLAGS) $< $(CORE_OBJS) -o $@ $(LDFLAGS)

$(TEST_BIN)/test_ipc_fd: tests/test_ipc_fd.c $(CORE_OBJS) | $(TEST_BIN)
	$(CC) $(CFLAGS) $< $(CORE_OBJS) -o $@ $(LDFLAGS)

$(TOOL_BIN)/kbox: tools/kbox.c | $(TOOL_BIN)
	$(CC) -Wall -Wextra -O2 -g $< -o $@

$(TOOL_BIN)/kbox_worker_daemon: tools/kbox_worker_daemon.c tools/kbox_protocol.h $(CORE_OBJS) | $(TOOL_BIN)
	$(CC) $(CFLAGS) $< $(CORE_OBJS) -o $@ $(LDFLAGS)

test: $(C_TESTS)
	@echo "=== Iterate Core Smoke Tests ==="
	@PASS=0; FAIL=0; \
	for t in $(C_TESTS); do \
		name=$$(basename $$t); \
		printf "  %-25s " "$$name"; \
		output=$$(timeout 30s $$t 2>&1); \
		rc=$$?; \
		if [ $$rc -eq 0 ]; then \
			echo "PASS"; PASS=$$((PASS+1)); \
		else \
			echo "FAIL (rc=$$rc)"; FAIL=$$((FAIL+1)); \
			echo "$$output" | head -20; \
		fi; \
	done; \
	echo "---"; \
	echo "$$PASS passed, $$FAIL failed"; \
	[ $$FAIL -eq 0 ]

test-cli: $(TOOLS)
	@bash tests/test_kbox_cli.sh

test-all: test test-cli

sync-python:
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync

install: all
	@echo "Installing kbox to $(PREFIX)..."
	install -d $(LIBEXECDIR)
	install -m 755 $(TOOL_BIN)/kbox_worker_daemon $(LIBEXECDIR)/
	install -m 755 tools/kbox_iterate.py $(LIBEXECDIR)/
	install -m 755 tools/kbox_isolated_kernel_mode.py $(LIBEXECDIR)/
	install -m 755 tools/kbox_mcp.py $(LIBEXECDIR)/
	install -d $(BINDIR)
	install -m 755 $(TOOL_BIN)/kbox $(BINDIR)/

uninstall:
	rm -f $(BINDIR)/kbox
	rm -rf $(LIBEXECDIR)

clean:
	rm -rf $(B)

compile_commands.json:
	@echo "Generating compile_commands.json..."
	@echo "[" > $@.tmp
	@sep=""; \
	for f in src/cuda_check.c src/ipc.c src/vmm.c tools/kbox.c tools/kbox_worker_daemon.c tests/test_cuda_check.c tests/test_ipc_fd.c; do \
		[ -f "$$f" ] || continue; \
		printf '%s{"directory":"%s","file":"%s","command":"%s %s %s"}' \
			"$$sep" "$$(pwd)" "$$f" "$(CC)" "$(CFLAGS)" "$$f" >> $@.tmp; \
		sep=","; \
	done; \
	echo "]" >> $@.tmp
	@mv $@.tmp $@
	@echo "  -> compile_commands.json"

-include $(DEPS)
