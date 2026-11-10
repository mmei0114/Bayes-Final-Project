.PHONY: all report check summarize verify

all report:
	$(MAKE) -C revision_v2 all

check:
	$(MAKE) -C revision_v2 check

summarize:
	$(MAKE) -C revision_v2 summarize

verify:
	shasum -a 256 -c MANIFEST.sha256
	cd data && shasum -a 256 -c SHA256SUMS
