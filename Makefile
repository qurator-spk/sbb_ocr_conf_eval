PIP_INSTALL = pip install

install:
	$(PIP_INSTALL) .

install-dev: PIP_INSTALL = pip install -e
install-dev: install
