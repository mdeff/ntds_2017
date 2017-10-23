DIRS = $(wildcard */)
CLEANDIRS = $(DIRS:%=clean-%)

run: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

clean: $(CLEANDIRS)

$(CLEANDIRS):
	$(MAKE) clean -C $(@:clean-%=%)

readme:
	grip README.md

.PHONY: run $(DIRS) clean $(CLEANDIRS) readme
